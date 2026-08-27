//! GpuQwen35 — modello ibrido residente su GPU (F7).
//!
//! Layout pesi: upload f32 una tantum (F16 golden), poi solo token IDS H2D.
//! State residente: DeltaNet → [S (d²·H) | conv_buf (K·D)] per 18 layer,
//! Attention → KV cache [layer][pos][kv_dim] per 6 layer.
//! Forward ibrido: DeltaNet loop token-by-token (recurrence), Attention batched.

use crate::runtime::{CudaCtx, err};
use cust::memory::{CopyDestination, DeviceBuffer, DevicePointer};
use cust::module::Module;
use s_models::qwen35::{Config, Model, Qwen35LayerView};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

macro_rules! launch_simple {
    ($func:expr, $stream:expr, $grid:expr, $block:expr, $($args:expr),* $(,)?) => {{
        let func = match $func {
            Ok(f) => f,
            Err(e) => panic!("launch (get_function): {e}"),
        };
        let g: u32 = $grid;
        let b: u32 = $block;
        let s = $stream;
        if let Err(e) = unsafe { cust::launch!(func<<<g, b, 0, s>>>($($args),*)) } {
            panic!("launch: {e}");
        }
    }};
}

fn grid_for(total: usize, block: u32) -> u32 { ((total as u32)+block-1)/block }
fn ptr_off(p: DevicePointer<f32>, elems: usize) -> DevicePointer<f32> {
    DevicePointer::from_raw(unsafe{ p.as_ptr().add(elems)} as u64)
}

struct GpuCore {
    _ctx: CudaCtx,
    module: Module,
    stream: cust::stream::Stream,
}
impl GpuCore {
    fn new() -> Result<Self, String> {
        let ctx = CudaCtx::new(0)?;
        let module = Module::from_cubin(CUBIN, &[]).map_err(|e| err("from_cubin", e))?;
        let stream = { use cust::stream::{Stream, StreamFlags}; Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| err("stream", e))? };
        Ok(Self{ _ctx: ctx, module, stream })
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
    ssm_norm: DeviceBuffer<f32>, // expanded to n_heads*hd (2048) if needed
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
    q_norm: DeviceBuffer<f32>, // expanded
    k_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    ffn_gate: DeviceBuffer<f32>,
    ffn_up: DeviceBuffer<f32>,
    ffn_down: DeviceBuffer<f32>,
}
enum GpuLayer35 { DeltaNet(GpuDeltaNetLayer), Attention(GpuAttentionLayer35) }

pub struct GpuQwen35Cache { pub(crate) len: usize }

pub struct GpuQwen35 {
    core: GpuCore,
    config: Config,
    tok_embd: DeviceBuffer<f32>,
    output_head_dev: DeviceBuffer<f32>,
    output_norm_dev: DeviceBuffer<f32>,
    output_norm_host: Vec<f32>,
    output_head_host: Vec<f32>,
    layers: Vec<GpuLayer35>,
    states: Vec<DeviceBuffer<f32>>,
    conv_bufs: Vec<DeviceBuffer<f32>>,
    k_caches: Vec<DeviceBuffer<f32>>,
    v_caches: Vec<DeviceBuffer<f32>>,
    max_seq: usize,
    // scratch batched
    x: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    q_full: DeviceBuffer<f32>,
    q_buf: DeviceBuffer<f32>,
    gate_buf: DeviceBuffer<f32>,
    k_new: DeviceBuffer<f32>,
    v_new: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    tmp1: DeviceBuffer<f32>,
    tmp2: DeviceBuffer<f32>,
    ffn_act: DeviceBuffer<f32>,
    // delta single-token scratch
    delta_qkv: DeviceBuffer<f32>,
    delta_conv_out: DeviceBuffer<f32>,
    delta_z: DeviceBuffer<f32>,
    delta_alpha_dot: DeviceBuffer<f32>,
    delta_beta_dot: DeviceBuffer<f32>,
    delta_g: DeviceBuffer<f32>,
    delta_beta: DeviceBuffer<f32>,
    delta_out: DeviceBuffer<f32>,
    last_hidden: DeviceBuffer<f32>,
}

fn up(data: &[f32]) -> Result<DeviceBuffer<f32>, String> { DeviceBuffer::from_slice(data).map_err(|e| err("upload", e)) }
fn up_expanded(data: &[f32], hd: usize, n_heads: usize) -> Result<DeviceBuffer<f32>, String> {
    if data.len() == hd * n_heads { return up(data); }
    if data.len() == hd {
        let mut expanded = Vec::with_capacity(hd * n_heads);
        for _ in 0..n_heads { expanded.extend_from_slice(data); }
        return up(&expanded);
    }
    // fallback: if mismatched, just upload as is (will error later if kernel reads OOB)
    up(data)
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
        let zero_f = |n: usize| -> Result<DeviceBuffer<f32>, String> { DeviceBuffer::zeroed(n).map_err(|e| err("alloc", e)) };
        for lw in views.layers.iter() {
            match lw {
                Qwen35LayerView::DeltaNet(dn) => {
                    layers.push(GpuLayer35::DeltaNet(GpuDeltaNetLayer{
                        attn_norm: up(dn.attn_norm)?,
                        qkv: up(dn.qkv)?,
                        gate: up(dn.gate)?,
                        conv1d: up(dn.conv1d)?,
                        dt_bias: up(dn.dt_bias)?,
                        ssm_a: up(dn.ssm_a)?,
                        alpha: up(dn.alpha)?,
                        beta: up(dn.beta)?,
                        ssm_norm: up_expanded(dn.ssm_norm, 128, n_heads)?,
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
                },
                Qwen35LayerView::Attention(at) => {
                    layers.push(GpuLayer35::Attention(GpuAttentionLayer35{
                        attn_norm: up(at.attn_norm)?,
                        wq: up(at.wq)?,
                        wk: up(at.wk)?,
                        wv: up(at.wv)?,
                        wo: up(at.wo)?,
                        q_norm: up_expanded(at.q_norm, config.head_dim, config.n_heads)?,
                        k_norm: up_expanded(at.k_norm, config.head_dim, config.n_kv_heads)?,
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
        let q_dim = config.n_heads * config.head_dim;
        let q_full_dim = q_dim * 2;
        // scores size: worst max_seq*max_seq*n_heads (like GpuQwen2) — capped for large max_seq
        let scores_elems = max_seq * max_seq * config.n_heads;
        Ok(Self{
            core,
            tok_embd: up(views.tok_embd)?,
            output_head_dev: up(views.output)?,
            output_norm_dev: up(views.output_norm)?,
            output_norm_host: views.output_norm.to_vec(),
            output_head_host: views.output.to_vec(),
            layers, states, conv_bufs, k_caches, v_caches,
            max_seq,
            x: zero_f(max_seq * ne)?,
            normed: zero_f(max_seq * ne)?,
            q_full: zero_f(max_seq * q_full_dim)?,
            q_buf: zero_f(max_seq * q_dim)?,
            gate_buf: zero_f(max_seq * q_dim)?,
            k_new: zero_f(max_seq * kv_dim)?,
            v_new: zero_f(max_seq * kv_dim)?,
            attn_out: zero_f(max_seq * q_dim)?,
            scores: zero_f(scores_elems)?,
            tmp1: zero_f(max_seq * n_ff.max(ne))?,
            tmp2: zero_f(max_seq * n_ff.max(ne))?,
            ffn_act: zero_f(max_seq * n_ff)?,
            delta_qkv: zero_f(conv_dim)?,
            delta_conv_out: zero_f(conv_dim)?,
            delta_z: zero_f(value_dim)?,
            delta_alpha_dot: zero_f(n_heads)?,
            delta_beta_dot: zero_f(n_heads)?,
            delta_g: zero_f(n_heads)?,
            delta_beta: zero_f(n_heads)?,
            delta_out: zero_f(value_dim)?,
            last_hidden: zero_f(ne)?,
            config,
        })
    }
    pub fn describe(&self) -> Result<String,String>{ self.core._ctx.describe() }

    fn forward(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let (ne, hd) = (c.n_embd, c.head_dim);
        let n_heads = c.n_heads;
        let n_kv_heads = c.n_kv_heads;
        let kv_dim = n_kv_heads * hd;
        let q_dim = n_heads * hd;
        let q_full_dim = q_dim * 2;
        let n_ff = c.n_ff;
        let seq = tokens.len();
        if seq==0 { return Err("forward: nessun token".into()); }
        if pos0+seq > self.max_seq { return Err(format!("seq oltre max_seq {}+{}>{}", pos0, seq, self.max_seq)); }
        let stream = &self.core.stream;
        let key_dim = c.ssm_d_state * c.ssm_n_group;
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;
        let conv_dim = key_dim*2+value_dim;
        let d_state = c.ssm_d_state;
        let n_v_heads = c.ssm_dt_rank;
        let conv_k = c.ssm_d_conv;
        let scale = 1.0/(hd as f32).sqrt();

        // embedding lookup
        let d_ids = DeviceBuffer::from_slice(tokens).map_err(|e| err("upload ids", e))?;
        launch_simple!(self.core.function("gather_rows_f32"), stream, grid_for(seq*ne,256),256, self.tok_embd.as_device_ptr(), d_ids.as_device_ptr(), self.x.as_device_ptr(), seq as u32, ne as u32);

        // Se pos0==0, nuova sequenza: azzera stato ricorrente e KV cache (erano sporchi da sequenza precedente sul medesimo modello globale)
        if pos0 == 0 {
            for buf in &self.states {
                if buf.len() > 1 {
                    launch_simple!(self.core.function("zero_f32"), stream, grid_for(buf.len(),256),256, buf.as_device_ptr(), buf.len() as u32);
                }
            }
            for buf in &self.conv_bufs {
                if buf.len() > 1 {
                    launch_simple!(self.core.function("zero_f32"), stream, grid_for(buf.len(),256),256, buf.as_device_ptr(), buf.len() as u32);
                }
            }
            for buf in &self.k_caches { if buf.len() > 1 { launch_simple!(self.core.function("zero_f32"), stream, grid_for(buf.len(),256),256, buf.as_device_ptr(), buf.len() as u32); } }
            for buf in &self.v_caches { if buf.len() > 1 { launch_simple!(self.core.function("zero_f32"), stream, grid_for(buf.len(),256),256, buf.as_device_ptr(), buf.len() as u32); } }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                GpuLayer35::DeltaNet(dn) => {
                    // sequential per token
                    for s in 0..seq {
                        let x_ptr = ptr_off(self.x.as_device_ptr(), s*ne);
                        let normed_ptr = ptr_off(self.normed.as_device_ptr(), s*ne);
                        // 1 rmsnorm 1xne
                        launch_simple!(self.core.function("rmsnorm_f32"), stream, 1,64, x_ptr, dn.attn_norm.as_device_ptr(), normed_ptr, 1, ne as u32, c.rms_eps);
                        // 2 qkv & gate matmuls seq=1
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(conv_dim,256),256, normed_ptr, dn.qkv.as_device_ptr(), self.delta_qkv.as_device_ptr(), 1, conv_dim as u32, ne as u32);
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(value_dim,256),256, normed_ptr, dn.gate.as_device_ptr(), self.delta_z.as_device_ptr(), 1, value_dim as u32, ne as u32);
                        // 3 conv1d depthwise
                        launch_simple!(self.core.function("conv1d_depthwise_f32"), stream, grid_for(conv_dim,256),256, self.conv_bufs[li].as_device_ptr(), self.delta_qkv.as_device_ptr(), dn.conv1d.as_device_ptr(), self.delta_conv_out.as_device_ptr(), conv_k as u32, conv_dim as u32);
                        // 4 silu inplace
                        launch_simple!(self.core.function("silu_inplace_f32"), stream, grid_for(conv_dim,256),256, self.delta_conv_out.as_device_ptr(), conv_dim as u32);
                        // 5 l2norm per head Q & K
                        let q_ptr = self.delta_conv_out.as_device_ptr();
                        let k_ptr = ptr_off(self.delta_conv_out.as_device_ptr(), key_dim);
                        launch_simple!(self.core.function("l2norm_per_head_f32"), stream, grid_for(n_v_heads,64),64, q_ptr, n_v_heads as u32, d_state as u32);
                        launch_simple!(self.core.function("l2norm_per_head_f32"), stream, grid_for(n_v_heads,64),64, k_ptr, n_v_heads as u32, d_state as u32);
                        let q_scale = 1.0f32 / (d_state as f32).sqrt();
                        launch_simple!(self.core.function("scale_f32"), stream, grid_for(key_dim,256),256, q_ptr, key_dim as u32, q_scale);
                        // 6 alpha & beta matmuls
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(n_v_heads,64),64, normed_ptr, dn.alpha.as_device_ptr(), self.delta_alpha_dot.as_device_ptr(), 1, n_v_heads as u32, ne as u32);
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(n_v_heads,64),64, normed_ptr, dn.beta.as_device_ptr(), self.delta_beta_dot.as_device_ptr(), 1, n_v_heads as u32, ne as u32);
                        launch_simple!(self.core.function("delta_gate_f32"), stream, grid_for(n_v_heads,64),64, self.delta_alpha_dot.as_device_ptr(), dn.dt_bias.as_device_ptr(), dn.ssm_a.as_device_ptr(), self.delta_g.as_device_ptr(), n_v_heads as u32);
                        launch_simple!(self.core.function("delta_beta_f32"), stream, grid_for(n_v_heads,64),64, self.delta_beta_dot.as_device_ptr(), self.delta_beta.as_device_ptr(), n_v_heads as u32);
                        // 7 delta recurrent step
                        let v_ptr = ptr_off(self.delta_conv_out.as_device_ptr(), 2*key_dim);
                        launch_simple!(self.core.function("delta_recurrent_step_f32"), stream, grid_for(n_v_heads,64),64, self.states[li].as_device_ptr(), q_ptr, k_ptr, v_ptr, self.delta_g.as_device_ptr(), self.delta_beta.as_device_ptr(), self.delta_out.as_device_ptr(), n_v_heads as u32, d_state as u32, d_state as u32);
                        // 8 gated norm per head
                        launch_simple!(self.core.function("rmsnorm_per_head_f32"), stream, grid_for(n_v_heads,64),64, self.delta_out.as_device_ptr(), dn.ssm_norm.as_device_ptr(), n_v_heads as u32, d_state as u32, c.rms_eps);
                        launch_simple!(self.core.function("silu_mul_f32"), stream, grid_for(value_dim,256),256, self.delta_z.as_device_ptr(), self.delta_out.as_device_ptr(), self.delta_out.as_device_ptr(), value_dim as u32);
                        // 9 ssm_out matmul -> tmp
                        // Use tmp1 as single token output [ne]
                        let tmp_ptr = self.tmp1.as_device_ptr();
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(ne,256),256, self.delta_out.as_device_ptr(), dn.ssm_out.as_device_ptr(), tmp_ptr, 1, ne as u32, value_dim as u32);
                        // 10 residual add x += tmp
                        launch_simple!(self.core.function("vec_add_f32"), stream, grid_for(ne,256),256, x_ptr, tmp_ptr, x_ptr, ne as u32);
                        // 11 post_norm + FFN per token
                        // post_norm
                        let post_tmp = ptr_off(self.tmp1.as_device_ptr(), 0); // reuse
                        // Need normed for FFN: we can reuse normed_ptr already? But need post_normed buffer: use tmp2 row?
                        // Use tmp2 as post_normed single token [ne]
                        let post_normed_ptr = self.tmp2.as_device_ptr(); // [ne]
                        launch_simple!(self.core.function("rmsnorm_f32"), stream, 1,64, x_ptr, dn.post_norm.as_device_ptr(), post_normed_ptr, 1, ne as u32, c.rms_eps);
                        // gate & up
                        let gate_ptr = self.tmp1.as_device_ptr(); // reuse tmp1 for gate [n_ff]
                        let up_ptr = ptr_off(self.tmp2.as_device_ptr(), ne); // offset after post_normed
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(n_ff,256),256, post_normed_ptr, dn.ffn_gate.as_device_ptr(), gate_ptr, 1, n_ff as u32, ne as u32);
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(n_ff,256),256, post_normed_ptr, dn.ffn_up.as_device_ptr(), up_ptr, 1, n_ff as u32, ne as u32);
                        launch_simple!(self.core.function("silu_mul_f32"), stream, grid_for(n_ff,256),256, gate_ptr, up_ptr, self.ffn_act.as_device_ptr(), n_ff as u32);
                        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(ne,256),256, self.ffn_act.as_device_ptr(), dn.ffn_down.as_device_ptr(), post_normed_ptr, 1, ne as u32, n_ff as u32);
                        launch_simple!(self.core.function("vec_add_f32"), stream, grid_for(ne,256),256, x_ptr, post_normed_ptr, x_ptr, ne as u32);
                    }
                },
                GpuLayer35::Attention(at) => {
                    // batched attention
                    // rmsnorm
                    launch_simple!(self.core.function("rmsnorm_f32"), stream, seq as u32,64, self.x.as_device_ptr(), at.attn_norm.as_device_ptr(), self.normed.as_device_ptr(), seq as u32, ne as u32, c.rms_eps);
                    // q_full = normed @ wq^T  [seq,4096]
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*q_full_dim,256),256, self.normed.as_device_ptr(), at.wq.as_device_ptr(), self.q_full.as_device_ptr(), seq as u32, q_full_dim as u32, ne as u32);
                    // split q/gate
                    launch_simple!(self.core.function("split_q_gate_f32"), stream, grid_for(seq*q_dim,256),256, self.q_full.as_device_ptr(), self.q_buf.as_device_ptr(), self.gate_buf.as_device_ptr(), seq as u32);
                    // q rmsnorm per head — need per seq batched: we have kernel rmsnorm_per_head for single token (n_heads). For batched we need to loop per token or extend kernel. For now loop per token s.
                    for s in 0..seq {
                        let q_s = ptr_off(self.q_buf.as_device_ptr(), s*q_dim);
                        launch_simple!(self.core.function("rmsnorm_per_head_f32"), stream, grid_for(n_heads,64),64, q_s, at.q_norm.as_device_ptr(), n_heads as u32, hd as u32, c.rms_eps);
                    }
                    // k & v matmuls
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*kv_dim,256),256, self.normed.as_device_ptr(), at.wk.as_device_ptr(), self.k_new.as_device_ptr(), seq as u32, kv_dim as u32, ne as u32);
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*kv_dim,256),256, self.normed.as_device_ptr(), at.wv.as_device_ptr(), self.v_new.as_device_ptr(), seq as u32, kv_dim as u32, ne as u32);
                    for s in 0..seq {
                        let k_s = ptr_off(self.k_new.as_device_ptr(), s*kv_dim);
                        launch_simple!(self.core.function("rmsnorm_per_head_f32"), stream, grid_for(n_kv_heads,64),64, k_s, at.k_norm.as_device_ptr(), n_kv_heads as u32, hd as u32, c.rms_eps);
                    }
                    // rope mrope for q and k
                    launch_simple!(self.core.function("rope_mrope_f32"), stream, grid_for(seq*n_heads*(c.rope_n_rot/2),64),64, self.q_buf.as_device_ptr(), seq as u32, n_heads as u32, hd as u32, c.rope_theta, pos0 as u32, c.rope_n_rot as u32);
                    launch_simple!(self.core.function("rope_mrope_f32"), stream, grid_for(seq*n_kv_heads*(c.rope_n_rot/2),64),64, self.k_new.as_device_ptr(), seq as u32, n_kv_heads as u32, hd as u32, c.rope_theta, pos0 as u32, c.rope_n_rot as u32);
                    // append to cache at pos0
                    let kv_layer_off = pos0 * kv_dim;
                    launch_simple!(self.core.function("copy_n_f32"), stream, grid_for(seq*kv_dim,256),256, self.k_new.as_device_ptr(), ptr_off(self.k_caches[li].as_device_ptr(), kv_layer_off), (seq*kv_dim) as u32);
                    launch_simple!(self.core.function("copy_n_f32"), stream, grid_for(seq*kv_dim,256),256, self.v_new.as_device_ptr(), ptr_off(self.v_caches[li].as_device_ptr(), kv_layer_off), (seq*kv_dim) as u32);
                    // attention scores
                    let width = (pos0+seq) as u32;
                    launch_simple!(self.core.function("attention_scores_f32"), stream, grid_for(seq*n_heads,64),64, self.q_buf.as_device_ptr(), self.k_caches[li].as_device_ptr(), self.scores.as_device_ptr(), seq as u32, pos0 as u32, width, n_heads as u32, n_kv_heads as u32, hd as u32, kv_dim as u32, scale);
                    launch_simple!(self.core.function("softmax_rows_f32"), stream, grid_for(seq*n_heads,32),32, self.scores.as_device_ptr(), (seq*n_heads) as u32, width);
                    launch_simple!(self.core.function("attn_weighted_sum_f32"), stream, grid_for(seq*q_dim,256),256, self.scores.as_device_ptr(), self.v_caches[li].as_device_ptr(), self.attn_out.as_device_ptr(), seq as u32, pos0 as u32, width, n_heads as u32, n_kv_heads as u32, hd as u32, kv_dim as u32);
                    // gate sigmoid mul
                    launch_simple!(self.core.function("gate_sigmoid_mul_f32"), stream, grid_for(seq*q_dim,256),256, self.attn_out.as_device_ptr(), self.gate_buf.as_device_ptr(), (seq*q_dim) as u32);
                    // wo projection + residual
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*ne,256),256, self.attn_out.as_device_ptr(), at.wo.as_device_ptr(), self.tmp1.as_device_ptr(), seq as u32, ne as u32, q_dim as u32);
                    launch_simple!(self.core.function("vec_add_f32"), stream, grid_for(seq*ne,256),256, self.x.as_device_ptr(), self.tmp1.as_device_ptr(), self.x.as_device_ptr(), (seq*ne) as u32);
                    // FFN
                    launch_simple!(self.core.function("rmsnorm_f32"), stream, seq as u32,64, self.x.as_device_ptr(), at.post_norm.as_device_ptr(), self.normed.as_device_ptr(), seq as u32, ne as u32, c.rms_eps);
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*n_ff,256),256, self.normed.as_device_ptr(), at.ffn_gate.as_device_ptr(), self.tmp1.as_device_ptr(), seq as u32, n_ff as u32, ne as u32);
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*n_ff,256),256, self.normed.as_device_ptr(), at.ffn_up.as_device_ptr(), self.tmp2.as_device_ptr(), seq as u32, n_ff as u32, ne as u32);
                    launch_simple!(self.core.function("silu_mul_f32"), stream, grid_for(seq*n_ff,256),256, self.tmp1.as_device_ptr(), self.tmp2.as_device_ptr(), self.ffn_act.as_device_ptr(), (seq*n_ff) as u32);
                    launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(seq*ne,256),256, self.ffn_act.as_device_ptr(), at.ffn_down.as_device_ptr(), self.tmp2.as_device_ptr(), seq as u32, ne as u32, n_ff as u32);
                    launch_simple!(self.core.function("vec_add_f32"), stream, grid_for(seq*ne,256),256, self.x.as_device_ptr(), self.tmp2.as_device_ptr(), self.x.as_device_ptr(), (seq*ne) as u32);
                }
            }
        }
        // final norm + head
        let last_off = (seq-1)*ne;
        launch_simple!(self.core.function("rmsnorm_f32"), stream, 1,64, ptr_off(self.x.as_device_ptr(), last_off), self.output_norm_dev.as_device_ptr(), self.last_hidden.as_device_ptr(), 1, ne as u32, c.rms_eps);
        let logits_dev = DeviceBuffer::<f32>::zeroed(c.vocab_size).map_err(|e| err("alloc", e))?;
        launch_simple!(self.core.function("matmul_wt_f32"), stream, grid_for(c.vocab_size,256),256, self.last_hidden.as_device_ptr(), self.output_head_dev.as_device_ptr(), logits_dev.as_device_ptr(), 1, c.vocab_size as u32, ne as u32);
        stream.synchronize().map_err(|e| err("sync", e))?;
        let mut logits = vec![0.0f32; c.vocab_size];
        logits_dev.copy_to(&mut logits).map_err(|e| err("download", e))?;
        Ok(logits)
    }
}

impl s_models::Device for GpuQwen35 {
    type Cache = GpuQwen35Cache;
    type Config = Config;
    fn config(&self) -> &Config { &self.config }
    fn new_cache(&self, _max_seq: usize) -> GpuQwen35Cache { GpuQwen35Cache{ len:0 } }
    fn forward_cached(&self, tokens: &[u32], cache: &mut GpuQwen35Cache) -> Vec<f32> {
        match self.forward(tokens, cache.len) {
            Ok(l) => { cache.len += tokens.len(); l },
            Err(e) => panic!("gpu forward: {e}"),
        }
    }
}
