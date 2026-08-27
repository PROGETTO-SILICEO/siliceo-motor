//! GpuQwen35 — modello ibrido residente su GPU (F7).
//!
//! Layout pesi: upload f32 una tantum (F16 golden), poi solo token IDS H2D.
//! State residente: DeltaNet → [S (d²·H) | conv_buf (K·D)] per 18 layer,
//! Attention → KV cache [layer][pos][kv_dim] per 6 layer.

use crate::runtime::{CudaCtx, err};
use cust::memory::{DeviceBuffer, DevicePointer};
use cust::module::Module;
use s_models::qwen35::{Config, Model, Qwen35LayerView};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

struct GpuCore {
    _ctx: CudaCtx,
    module: Module,
    stream: cust::stream::Stream,
}

impl GpuCore {
    fn new() -> Result<Self, String> {
        let ctx = CudaCtx::new(0)?;
        let module = Module::from_cubin(CUBIN, &[]).map_err(|e| err("from_cubin", e))?;
        let stream = {
            use cust::stream::{Stream, StreamFlags};
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| err("stream", e))?
        };
        Ok(Self { _ctx: ctx, module, stream })
    }
    fn function(&self, name: &str) -> Result<cust::function::Function<'_>, String> {
        self.module.get_function(name).map_err(|e| err("get_function", e))
    }
}

struct GpuDeltaNetLayer {
    attn_norm: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    conv1d: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    ssm_a: DeviceBuffer<f32>,
    alpha: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    ssm_norm: DeviceBuffer<f32>,
    ssm_out: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    ffn_gate: DeviceBuffer<f32>,
    ffn_up: DeviceBuffer<f32>,
    ffn_down: DeviceBuffer<f32>,
}

struct GpuAttentionLayer35 {
    attn_norm: DeviceBuffer<f32>,
    wq: DeviceBuffer<f32>,
    wk: DeviceBuffer<f32>,
    wv: DeviceBuffer<f32>,
    wo: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    ffn_gate: DeviceBuffer<f32>,
    ffn_up: DeviceBuffer<f32>,
    ffn_down: DeviceBuffer<f32>,
}

enum GpuLayer35 {
    DeltaNet(GpuDeltaNetLayer),
    Attention(GpuAttentionLayer35),
}

pub struct GpuQwen35Cache {
    pub(crate) len: usize,
}

pub struct GpuQwen35 {
    core: GpuCore,
    config: Config,
    tok_embd: DeviceBuffer<f32>,
    output_head_dev: DeviceBuffer<f32>,
    output_norm_dev: DeviceBuffer<f32>,
    output_norm_host: Vec<f32>,
    output_head_host: Vec<f32>,
    layers: Vec<GpuLayer35>,
    // state residente
    states: Vec<DeviceBuffer<f32>>,      // per DeltaNet: [H·d²]
    conv_bufs: Vec<DeviceBuffer<f32>>,   // per DeltaNet: [K·D]
    k_caches: Vec<DeviceBuffer<f32>>,    // per Attention: [max_seq·kv_dim] o zeroed se DeltaNet
    v_caches: Vec<DeviceBuffer<f32>>,
    max_seq: usize,
    // scratch
    x: DeviceBuffer<f32>,
    tmp1: DeviceBuffer<f32>,
    tmp2: DeviceBuffer<f32>,
}

fn up(data: &[f32]) -> Result<DeviceBuffer<f32>, String> {
    DeviceBuffer::from_slice(data).map_err(|e| err("upload", e))
}

impl GpuQwen35 {
    pub fn load(cpu_model: &Model, max_seq: usize) -> Result<Self, String> {
        let config = cpu_model.config.clone();
        let ne = config.n_embd;
        let kv_dim = config.n_kv_heads * config.head_dim;
        let key_dim = config.ssm_d_state * config.ssm_n_group;
        let value_dim = config.ssm_d_state * config.ssm_dt_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let d_state = config.ssm_d_state;
        let n_heads = config.ssm_dt_rank;

        let core = GpuCore::new()?;
        let views = cpu_model.weight_views_for_gpu();

        let mut layers = Vec::with_capacity(config.n_layers);
        let mut states = Vec::with_capacity(config.n_layers);
        let mut conv_bufs = Vec::with_capacity(config.n_layers);
        let mut k_caches = Vec::with_capacity(config.n_layers);
        let mut v_caches = Vec::with_capacity(config.n_layers);

        let zero_f = |n: usize| -> Result<DeviceBuffer<f32>, String> {
            DeviceBuffer::zeroed(n).map_err(|e| err("alloc", e))
        };

        for lw in views.layers.iter() {
            match lw {
                Qwen35LayerView::DeltaNet(dn) => {
                    layers.push(GpuLayer35::DeltaNet(GpuDeltaNetLayer {
                        attn_norm: up(dn.attn_norm)?,
                        qkv: up(dn.qkv)?,
                        gate: up(dn.gate)?,
                        conv1d: up(dn.conv1d)?,
                        dt_bias: up(dn.dt_bias)?,
                        ssm_a: up(dn.ssm_a)?,
                        alpha: up(dn.alpha)?,
                        beta: up(dn.beta)?,
                        ssm_norm: up(dn.ssm_norm)?,
                        ssm_out: up(dn.ssm_out)?,
                        post_norm: up(dn.post_norm)?,
                        ffn_gate: up(dn.ffn_gate)?,
                        ffn_up: up(dn.ffn_up)?,
                        ffn_down: up(dn.ffn_down)?,
                    }));
                    states.push(zero_f(d_state * d_state * n_heads)?);
                    conv_bufs.push(zero_f(config.ssm_d_conv * conv_dim)?);
                    k_caches.push(zero_f(1)?);
                    v_caches.push(zero_f(1)?);
                }
                Qwen35LayerView::Attention(at) => {
                    layers.push(GpuLayer35::Attention(GpuAttentionLayer35 {
                        attn_norm: up(at.attn_norm)?,
                        wq: up(at.wq)?,
                        wk: up(at.wk)?,
                        wv: up(at.wv)?,
                        wo: up(at.wo)?,
                        q_norm: up(at.q_norm)?,
                        k_norm: up(at.k_norm)?,
                        post_norm: up(at.post_norm)?,
                        ffn_gate: up(at.ffn_gate)?,
                        ffn_up: up(at.ffn_up)?,
                        ffn_down: up(at.ffn_down)?,
                    }));
                    states.push(zero_f(1)?);
                    conv_bufs.push(zero_f(1)?);
                    k_caches.push(zero_f(max_seq * kv_dim)?);
                    v_caches.push(zero_f(max_seq * kv_dim)?);
                }
            }
        }

        let n_ff = config.n_ff;
        Ok(Self {
            core,
            tok_embd: up(views.tok_embd)?,
            output_head_dev: up(views.output)?,
            output_norm_dev: up(views.output_norm)?,
            output_norm_host: views.output_norm.to_vec(),
            output_head_host: views.output.to_vec(),
            layers,
            states,
            conv_bufs,
            k_caches,
            v_caches,
            max_seq,
            x: zero_f(max_seq * ne)?,
            tmp1: zero_f(max_seq * n_ff.max(ne))?,
            tmp2: zero_f(max_seq * n_ff.max(ne))?,
            config,
        })
    }

    pub fn describe(&self) -> Result<String, String> {
        self.core._ctx.describe()
    }
}

impl s_models::Device for GpuQwen35 {
    type Cache = GpuQwen35Cache;
    type Config = Config;
    fn config(&self) -> &Config { &self.config }
    fn new_cache(&self, _max_seq: usize) -> GpuQwen35Cache { GpuQwen35Cache { len: 0 } }
    fn forward_cached(&self, _tokens: &[u32], _cache: &mut GpuQwen35Cache) -> Vec<f32> {
        todo!("GpuQwen35 forward_cached — F7-3")
    }
}
