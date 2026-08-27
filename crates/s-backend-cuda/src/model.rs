//! Modello qwen2 residente su GPU — F3b step 4.
//!
//! Architettura (parity-first, decisioni roadmap v2):
//! - pesi dequantizzati f32 upload UNA TANTUM al load: nel decode le uniche
//!   copie H2D sono gli id dei token; solo l'ultima hidden state torna a host;
//! - KV cache su device `[layer][pos][kv_dim]`: stessa semantica
//!   posizioni-assolute di `s_models::SeqState`, ma i buffer non escono mai
//!   dalla GPU. Il `Cache` del trait resta quindi solo il contatore `len`;
//! - forward = sequenza di launch sugli STESSI kernel verificati con parity
//!   primitiva (matmul/rmsnorm/rope/softmax/silu/add) + kernel nuovi
//!   (gather/bias/attention) con le LORO parity;
//! - ordine di accumulo identico al CPU per costruzione (riduzioni
//!   sequenziali dentro un thread): parity attesa quasi bit-exact;
//! - ultimo stadio (norm finale + head) CONDIVISO col CPU tramite la free
//!   function `s_models::qwen2::logits_from_hidden`: una sola implementazione.
//!
//! Vincoli documentati:
//! - scratch pre-allocato su max_seq (scores ~ seq×W×heads: prompt lunghi =
//!   memoria dominante → fused attention nel tuning);
//! - 9B f32 NON entra in 24GB: questo path è per il 0.5B e la verifica;
//!   fused dequant-matmul = F3c;
//! - copie host di output_norm/output head (~0.5GB sul 0.5B): servono allo
//!   stadio finale condiviso; host RAM abbondante, accettabile ora.
//!
//! NOTA drop-order: `core` è il primo campo, stesso assetto dei smoke test
//! verdi su GPU (l'ordine di drop dei buffer cust segue la dichiarazione).

// macro testuale: DEVE precedere l'uso nel file (scoping testuale).
// Versione panicking: un launch fallito è fatale per il forward (kernel
// mancante o argomenti sbagliati), non c'è recovery sensata.
macro_rules! launch_simple {
    ($func:expr, $stream:expr, $grid:expr, $block:expr, $($args:expr),* $(,)?) => {{
        let func = match $func {
            Ok(f) => f,
            Err(e) => panic!("launch (get_function): {e}"),
        };
        let g: u32 = $grid;
        let b: u32 = $block;
        let s = $stream;
        if let Err(e) = unsafe {
            cust::launch!(func<<<g, b, 0, s>>>($($args),*))
        } {
            panic!("launch: {e}");
        }
    }};
}

use crate::runtime::{CudaCtx, err};
use cust::memory::{CopyDestination, DeviceBuffer, DevicePointer};
use cust::module::Module;
use s_models::qwen2::Config;
use s_models::{Device, Model};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

/// Contesto + modulo + stream: deve restare vivo finché vivono i buffer.
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

/// Pesi di un layer sui device buffer.
struct GpuLayer {
    attn_norm: DeviceBuffer<f32>,
    wq: DeviceBuffer<f32>,
    bq: Option<DeviceBuffer<f32>>,
    wk: DeviceBuffer<f32>,
    bk: Option<DeviceBuffer<f32>>,
    wv: DeviceBuffer<f32>,
    bv: Option<DeviceBuffer<f32>>,
    wo: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    w_gate: DeviceBuffer<f32>,
    w_up: DeviceBuffer<f32>,
    w_down: DeviceBuffer<f32>,
}

/// Cache del trait: K/V vivono su device (mai scaricate), qui resta solo
/// la posizione assoluta della sequenza.
pub struct GpuCache {
    pub(crate) len: usize,
}

impl GpuCache {
    pub fn len(&self) -> usize {
        self.len
    }
}

pub struct GpuQwen2 {
    core: GpuCore,
    config: Config,
    // ── pesi residenti su device ──
    tok_embd: DeviceBuffer<f32>,
    output_head_dev: DeviceBuffer<f32>,
    output_norm_dev: DeviceBuffer<f32>,
    layers: Vec<GpuLayer>,
    // ── copie host per lo stadio finale condiviso ──
    output_norm_host: Vec<f32>,
    output_head_host: Vec<f32>,
    // ── KV cache device [layer][pos][kv_dim] ──
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
    max_seq: usize,
    // ── scratch pre-allocato ──
    x: DeviceBuffer<f32>,       // hidden states [max_seq × ne]
    normed: DeviceBuffer<f32>,
    q_buf: DeviceBuffer<f32>,
    k_new: DeviceBuffer<f32>,
    v_new: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    tmp1: DeviceBuffer<f32>,    // max(ne,n_ff) × max_seq
    tmp2: DeviceBuffer<f32>,
    ffn_act: DeviceBuffer<f32>, // n_ff × max_seq
    scores: DeviceBuffer<f32>,  // worst case max_seq × W × n_heads
    last_hidden: DeviceBuffer<f32>,
}

/// Grid monodimensionale per `total` elementi con blocchi da `block`.
fn grid_for(total: usize, block: u32) -> u32 {
    ((total as u32) + block - 1) / block
}

impl GpuQwen2 {
    /// Carica i pesi del modello CPU sulla GPU e alloca cache+scratch.
    ///
    /// `cpu_model` serve solo durante il load (upload); dopo può essere
    /// rilasciato: GpuQwen2 è autosufficiente.
    pub fn load(cpu_model: &Model, max_seq: usize) -> Result<Self, String> {
        let config = cpu_model.config().clone();
        let (ne, n_ff) = (config.n_embd, config.n_ff);
        let kv_dim = config.n_kv_heads * config.head_dim;
        let views = cpu_model.weight_views();

        let core = GpuCore::new()?;

        let up = |data: &[f32]| -> Result<DeviceBuffer<f32>, String> {
            DeviceBuffer::from_slice(data).map_err(|e| err("upload", e))
        };
        let up_opt = |data: &Option<&[f32]>| -> Result<Option<DeviceBuffer<f32>>, String> {
            data.map(up).transpose()
        };

        let layers = views
            .layers
            .iter()
            .map(|l| {
                Ok(GpuLayer {
                    attn_norm: up(l.attn_norm)?,
                    wq: up(l.wq)?,
                    bq: up_opt(&l.bq)?,
                    wk: up(l.wk)?,
                    bk: up_opt(&l.bk)?,
                    wv: up(l.wv)?,
                    bv: up_opt(&l.bv)?,
                    wo: up(l.wo)?,
                    ffn_norm: up(l.ffn_norm)?,
                    w_gate: up(l.w_gate)?,
                    w_up: up(l.w_up)?,
                    w_down: up(l.w_down)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let zero_f = |n: usize| -> Result<DeviceBuffer<f32>, String> {
            DeviceBuffer::zeroed(n).map_err(|e| err("alloc", e))
        };

        Ok(Self {
            core,
            tok_embd: up(views.tok_embd)?,
            output_head_dev: up(views.output)?,
            output_norm_dev: up(views.output_norm)?,
            layers,
            // copie host dello stadio finale: UNA implementazione col CPU
            output_norm_host: views.output_norm.to_vec(),
            output_head_host: views.output.to_vec(),
            k_cache: zero_f(config.n_layers * max_seq * kv_dim)?,
            v_cache: zero_f(config.n_layers * max_seq * kv_dim)?,
            max_seq,
            x: zero_f(max_seq * ne)?,
            normed: zero_f(max_seq * ne)?,
            q_buf: zero_f(max_seq * ne)?,
            k_new: zero_f(max_seq * kv_dim)?,
            v_new: zero_f(max_seq * kv_dim)?,
            attn_out: zero_f(max_seq * ne)?,
            tmp1: zero_f(max_seq * ne.max(n_ff))?,
            tmp2: zero_f(max_seq * ne.max(n_ff))?,
            ffn_act: zero_f(max_seq * n_ff)?,
            scores: zero_f(max_seq * max_seq * config.n_heads)?,
            last_hidden: zero_f(ne)?,
            config,
        })
    }

    pub fn describe(&self) -> Result<String, String> {
        self.core._ctx.describe()
    }

    /// Forward incrementale: processa `tokens` da posizione assoluta `pos0`,
    /// ritorna i logits dell'ultimo token (host).
    pub fn forward(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let (ne, hd) = (c.n_embd, c.head_dim);
        let kv_dim = c.n_kv_heads * hd;
        let n_ff = c.n_ff;
        let seq = tokens.len();
        if seq == 0 {
            return Err("forward: nessun token".into());
        }
        if pos0 + seq > self.max_seq {
            return Err(format!("sequenza oltre max_seq: {} + {} > {}", pos0, seq, self.max_seq));
        }

        let stream = &self.core.stream;

        // ── embedding lookup (unica copia H2D del forward: gli id) ──
        let d_ids = DeviceBuffer::from_slice(tokens).map_err(|e| err("upload ids", e))?;
        launch_simple!(
            self.core.function("gather_rows_f32"),
            stream,
            grid_for(seq * ne, 256),
            256,
            self.tok_embd.as_device_ptr(),
            d_ids.as_device_ptr(),
            self.x.as_device_ptr(),
            seq as u32,
            ne as u32
        );

        let scale = 1.0 / (hd as f32).sqrt();

        for (li, layer) in self.layers.iter().enumerate() {
            // ── attenzione ──
            launch_simple!(
                self.core.function("rmsnorm_f32"),
                stream,
                seq as u32,
                64,
                self.x.as_device_ptr(),
                layer.attn_norm.as_device_ptr(),
                self.normed.as_device_ptr(),
                seq as u32,
                ne as u32,
                c.rms_eps
            );
            // q/k/v = normed @ W^T (+bias opzionale)
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.normed.as_device_ptr(),
                layer.wq.as_device_ptr(),
                self.q_buf.as_device_ptr(),
                seq as u32,
                ne as u32,
                ne as u32
            );
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.normed.as_device_ptr(),
                layer.wk.as_device_ptr(),
                self.k_new.as_device_ptr(),
                seq as u32,
                kv_dim as u32,
                ne as u32
            );
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.normed.as_device_ptr(),
                layer.wv.as_device_ptr(),
                self.v_new.as_device_ptr(),
                seq as u32,
                kv_dim as u32,
                ne as u32
            );
            let (bq_ptr, bq_has) = bias_args(&layer.bq, self.q_buf.as_device_ptr());
            launch_simple!(
                self.core.function("add_bias_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.q_buf.as_device_ptr(),
                bq_ptr,
                seq as u32,
                ne as u32,
                bq_has
            );
            let (bk_ptr, bk_has) = bias_args(&layer.bk, self.k_new.as_device_ptr());
            launch_simple!(
                self.core.function("add_bias_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.k_new.as_device_ptr(),
                bk_ptr,
                seq as u32,
                kv_dim as u32,
                bk_has
            );
            let (bv_ptr, bv_has) = bias_args(&layer.bv, self.v_new.as_device_ptr());
            launch_simple!(
                self.core.function("add_bias_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.v_new.as_device_ptr(),
                bv_ptr,
                seq as u32,
                kv_dim as u32,
                bv_has
            );
            // RoPE posizione assoluta
            launch_simple!(
                self.core.function("rope_neox_f32"),
                stream,
                grid_for(seq * c.n_heads * (hd / 2), 64),
                64,
                self.q_buf.as_device_ptr(),
                seq as u32,
                c.n_heads as u32,
                hd as u32,
                c.rope_theta,
                pos0 as u32
            );
            launch_simple!(
                self.core.function("rope_neox_f32"),
                stream,
                grid_for(seq * c.n_kv_heads * (hd / 2), 64),
                64,
                self.k_new.as_device_ptr(),
                seq as u32,
                c.n_kv_heads as u32,
                hd as u32,
                c.rope_theta,
                pos0 as u32
            );

            // append K/V nella cache device: layer li, posizione pos0
            let kv_layer_off = li * self.max_seq * kv_dim + pos0 * kv_dim;
            launch_simple!(
                self.core.function("copy_n_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.k_new.as_device_ptr(),
                ptr_off(self.k_cache.as_device_ptr(), kv_layer_off),
                (seq * kv_dim) as u32
            );
            launch_simple!(
                self.core.function("copy_n_f32"),
                stream,
                grid_for(seq * kv_dim, 256),
                256,
                self.v_new.as_device_ptr(),
                ptr_off(self.v_cache.as_device_ptr(), kv_layer_off),
                (seq * kv_dim) as u32
            );

            // punteggi causali (-INF oltre abs_s) → softmax larghezza W → Σw·v
            let width = (pos0 + seq) as u32;
            launch_simple!(
                self.core.function("attention_scores_f32"),
                stream,
                grid_for(seq * c.n_heads, 64),
                64,
                self.q_buf.as_device_ptr(),
                ptr_off(self.k_cache.as_device_ptr(), li * self.max_seq * kv_dim),
                self.scores.as_device_ptr(),
                seq as u32,
                pos0 as u32,
                width,
                c.n_heads as u32,
                c.n_kv_heads as u32,
                hd as u32,
                kv_dim as u32,
                scale
            );
            // UNA RIGA PER TESTA: [seq × heads × W]. Le posizioni paddate a
            // -INF danno exp=0 esatto: somma bitwise identica al CPU.
            launch_simple!(
                self.core.function("softmax_rows_f32"),
                stream,
                grid_for(seq * c.n_heads, 32),
                32,
                self.scores.as_device_ptr(),
                (seq * c.n_heads) as u32,
                width
            );
            launch_simple!(
                self.core.function("attn_weighted_sum_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.scores.as_device_ptr(),
                ptr_off(self.v_cache.as_device_ptr(), li * self.max_seq * kv_dim),
                self.attn_out.as_device_ptr(),
                seq as u32,
                pos0 as u32,
                width,
                c.n_heads as u32,
                c.n_kv_heads as u32,
                hd as u32,
                kv_dim as u32
            );
            // residuo: x += attn_out @ wo^T (vec_add alias-safe elemento per indice)
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.attn_out.as_device_ptr(),
                layer.wo.as_device_ptr(),
                self.tmp1.as_device_ptr(),
                seq as u32,
                ne as u32,
                ne as u32
            );
            launch_simple!(
                self.core.function("vec_add_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.x.as_device_ptr(),
                self.tmp1.as_device_ptr(),
                self.x.as_device_ptr(),
                (seq * ne) as u32
            );

            // ── FFN SwiGLU ──
            launch_simple!(
                self.core.function("rmsnorm_f32"),
                stream,
                seq as u32,
                64,
                self.x.as_device_ptr(),
                layer.ffn_norm.as_device_ptr(),
                self.normed.as_device_ptr(),
                seq as u32,
                ne as u32,
                c.rms_eps
            );
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * n_ff, 256),
                256,
                self.normed.as_device_ptr(),
                layer.w_gate.as_device_ptr(),
                self.tmp1.as_device_ptr(),
                seq as u32,
                n_ff as u32,
                ne as u32
            );
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * n_ff, 256),
                256,
                self.normed.as_device_ptr(),
                layer.w_up.as_device_ptr(),
                self.tmp2.as_device_ptr(),
                seq as u32,
                n_ff as u32,
                ne as u32
            );
            launch_simple!(
                self.core.function("silu_mul_f32"),
                stream,
                grid_for(seq * n_ff, 256),
                256,
                self.tmp1.as_device_ptr(),
                self.tmp2.as_device_ptr(),
                self.ffn_act.as_device_ptr(),
                (seq * n_ff) as u32
            );
            // down + residuo
            launch_simple!(
                self.core.function("matmul_wt_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.ffn_act.as_device_ptr(),
                layer.w_down.as_device_ptr(),
                self.tmp2.as_device_ptr(),
                seq as u32,
                ne as u32,
                n_ff as u32
            );
            launch_simple!(
                self.core.function("vec_add_f32"),
                stream,
                grid_for(seq * ne, 256),
                256,
                self.x.as_device_ptr(),
                self.tmp2.as_device_ptr(),
                self.x.as_device_ptr(),
                (seq * ne) as u32
            );
        }

        // ── finale: norm su device, head su device, download hidden+logits ──
        let last_off = (seq - 1) * ne;
        launch_simple!(
            self.core.function("rmsnorm_f32"),
            stream,
            1,
            64,
            ptr_off(self.x.as_device_ptr(), last_off),
            self.output_norm_dev.as_device_ptr(),
            self.last_hidden.as_device_ptr(),
            1,
            ne as u32,
            c.rms_eps
        );
        let logits_dev =
            DeviceBuffer::<f32>::zeroed(c.vocab_size).map_err(|e| err("alloc", e))?;
        launch_simple!(
            self.core.function("matmul_wt_f32"),
            stream,
            grid_for(c.vocab_size, 256),
            256,
            self.last_hidden.as_device_ptr(),
            self.output_head_dev.as_device_ptr(),
            logits_dev.as_device_ptr(),
            1,
            c.vocab_size as u32,
            ne as u32
        );

        stream.synchronize().map_err(|e| err("sync", e))?;

        let mut logits = vec![0.0f32; c.vocab_size];
        logits_dev.copy_to(&mut logits).map_err(|e| err("download", e))?;

        // verifica incrociata dello stadio condiviso: ricalcoliamo i logits
        // anche dall'hidden scaricato col codice CPU (una sola implementazione,
        // costo trascurabile: vocab × ne MAC su host). Attivabile per debug.
        if std::env::var_os("SILICEO_CUDA_CHECK_HEAD").is_some() {
            let mut hidden = vec![0.0f32; ne];
            self.last_hidden.copy_to(&mut hidden).map_err(|e| err("download", e))?;
            let ref_logits = Model::logits_from_hidden(
                &self.output_norm_host,
                &self.output_head_host,
                ne,
                c.rms_eps,
                &hidden,
            );
            let max_diff = logits
                .iter()
                .zip(ref_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if max_diff > 1e-3 {
                return Err(format!("stadio finale divergente: max_diff={max_diff}"));
            }
        }

        Ok(logits)
    }
}

/// Puntatore device offsettato di `elems` elementi f32.
/// cust espone DevicePointer (che implementa DeviceCopy), non puntatori
/// grezzi: si ricostruisce da raw dopo l'aritmetica.
fn ptr_off(p: DevicePointer<f32>, elems: usize) -> DevicePointer<f32> {
    DevicePointer::from_raw(unsafe { p.as_ptr().add(elems) } as u64)
}

/// Argomenti del kernel add_bias per un buffer opzionale: senza bias si
/// passa un puntatore dummy + flag 0 (il kernel esce subito). Evitiamo i
/// puntatori NULL attraverso la macro di launch (trait bound DeviceCopy).
fn bias_args(
    b: &Option<DeviceBuffer<f32>>,
    dummy: DevicePointer<f32>,
) -> (DevicePointer<f32>, u32) {
    match b {
        Some(buf) => (buf.as_device_ptr(), 1),
        None => (dummy, 0),
    }
}

/// Impl del trait: K/V su device, il Cache è solo il contatore di posizione.
impl Device for GpuQwen2 {
    type Cache = GpuCache;
    type Config = Config;

    fn config(&self) -> &Config {
        &self.config
    }

    fn new_cache(&self, _max_seq: usize) -> GpuCache {
        GpuCache { len: 0 }
    }

    fn forward_cached(&self, tokens: &[u32], cache: &mut GpuCache) -> Vec<f32> {
        match self.forward(tokens, cache.len) {
            Ok(logits) => {
                cache.len += tokens.len();
                logits
            }
            Err(e) => panic!("gpu forward: {e}"),
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Holder per i test: contesto+modulo vivi finché i buffer esistono.
    struct TestGpu(GpuCore);
    impl TestGpu {
        fn new() -> Self {
            Self(GpuCore::new().unwrap())
        }
    }

    /// Softmax per righe su buffer flat [rows × n]: replica del kernel,
    /// usata dal riferimento scores del test attention.
    fn softmax_rows_full(x: &mut [f32], rows: usize, n: usize) {
        for r in 0..rows {
            let row = &mut x[r * n..(r + 1) * n];
            let mut max = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    fn lcg_input(n: usize, seed: u64) -> Vec<f32> {
        let mut seed = seed;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| next()).collect()
    }

    /// Riferimento CPU del loop d'attenzione: replica ESATTA di
    /// qwen2::forward_cached (dot sequenziale su d, maschera ..=abs_s,
    /// softmax_row, Σw·v sequenziale su t).
    fn attn_cpu(
        q: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        seq: usize,
        pos0: usize,
        n_heads: usize,
        n_kv_heads: usize,
        hd: usize,
        kv_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let heads_per_kv = n_heads / n_kv_heads;
        let mut out = vec![0.0f32; seq * n_heads * hd];
        let mut scores = vec![0.0f32; pos0 + seq];
        for s in 0..seq {
            for h in 0..n_heads {
                let kvh = h / heads_per_kv;
                let abs_s = pos0 + s;
                for t in 0..=abs_s {
                    let q_off = (s * n_heads + h) * hd;
                    let k_off = t * kv_dim + kvh * hd;
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[q_off + d] * k_all[k_off + d];
                    }
                    scores[t] = dot * scale;
                }
                // softmax_row su ..=abs_s
                let row = &mut scores[..=abs_s];
                let mut max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max {
                        max = v;
                    }
                }
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
                let o_off = (s * n_heads + h) * hd;
                for d in 0..hd {
                    out[o_off + d] = 0.0;
                }
                for t in 0..=abs_s {
                    let w = scores[t];
                    let v_off = t * kv_dim + kvh * hd;
                    for d in 0..hd {
                        out[o_off + d] += w * v_all[v_off + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn attention_parity_contro_cpu_con_prefill_e_posizione_assoluta() {
        const SEQ: usize = 3;
        const POS0: usize = 7;
        const HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HD: usize = 8;
        const KV_DIM: usize = KV_HEADS * HD; // 16
        const MAX_SEQ: usize = 12;

        let gpu = TestGpu::new();
        let stream = &gpu.0.stream;

        let q = lcg_input(SEQ * HEADS * HD, 0xA77E);
        let mut k_all = lcg_input(MAX_SEQ * KV_DIM, 0xBEEF);
        let v_all = lcg_input(MAX_SEQ * KV_DIM, 0xCA5E5);

        // scala realistica: K/V ridotte per non saturare gli exp
        for v in k_all.iter_mut() {
            *v *= 0.3;
        }

        let width = POS0 + SEQ;
        let scale = 1.0 / (HD as f32).sqrt();
        let expected =
            attn_cpu(&q, &k_all, &v_all, SEQ, POS0, HEADS, KV_HEADS, HD, KV_DIM, scale);

        let d_q = DeviceBuffer::from_slice(&q).unwrap();
        let d_k = DeviceBuffer::from_slice(&k_all).unwrap();
        let d_v = DeviceBuffer::from_slice(&v_all).unwrap();
        let d_scores = DeviceBuffer::<f32>::zeroed(SEQ * HEADS * width).unwrap();
        let d_out = DeviceBuffer::<f32>::zeroed(SEQ * HEADS * HD).unwrap();

        // 1. punteggi
        launch_simple!(
            gpu.0.function("attention_scores_f32"),
            stream,
            grid_for(SEQ * HEADS, 64),
            64,
            d_q.as_device_ptr(),
            d_k.as_device_ptr(),
            d_scores.as_device_ptr(),
            SEQ as u32,
            POS0 as u32,
            width as u32,
            HEADS as u32,
            KV_HEADS as u32,
            HD as u32,
            KV_DIM as u32,
            scale
        );
        // 2. softmax su TUTTE le righe (una per testa): [seq × heads × W]
        launch_simple!(
            gpu.0.function("softmax_rows_f32"),
            stream,
            grid_for(SEQ * HEADS, 32),
            32,
            d_scores.as_device_ptr(),
            (SEQ * HEADS) as u32,
            width as u32
        );
        // 3. somma pesata
        launch_simple!(
            gpu.0.function("attn_weighted_sum_f32"),
            stream,
            grid_for(SEQ * HEADS * HD, 256),
            256,
            d_scores.as_device_ptr(),
            d_v.as_device_ptr(),
            d_out.as_device_ptr(),
            SEQ as u32,
            POS0 as u32,
            width as u32,
            HEADS as u32,
            KV_HEADS as u32,
            HD as u32,
            KV_DIM as u32
        );
        stream.synchronize().unwrap();

        // ── isolamento: verifica INTERMEDIA della matrice scores ──
        // Layout [seq × heads × W], riferimento CPU con la STESSA semantica
        // del path GPU (padding -INF, softmax a larghezza piena).
        let mut cpu_scores = vec![f32::NEG_INFINITY; SEQ * HEADS * width];
        for s in 0..SEQ {
            for h in 0..HEADS {
                let kvh = h / (HEADS / KV_HEADS);
                let abs_s = POS0 + s;
                for t in 0..=abs_s {
                    let q_off = (s * HEADS + h) * HD;
                    let k_off = t * KV_DIM + kvh * HD;
                    let mut dot = 0.0f32;
                    for d in 0..HD {
                        dot += q[q_off + d] * k_all[k_off + d];
                    }
                    cpu_scores[(s * HEADS + h) * width + t] = dot * scale;
                }
            }
        }
        softmax_rows_full(&mut cpu_scores, SEQ * HEADS, width);

        let got_scores = {
            let mut v = vec![0.0f32; SEQ * HEADS * width];
            d_scores.copy_to(&mut v).unwrap();
            v
        };
        let mut score_err = 0.0f32;
        for (g, e) in got_scores.iter().zip(cpu_scores.iter()) {
            if !e.is_infinite() || !g.is_infinite() {
                score_err = score_err.max((g - e).abs());
            }
        }
        println!("scores max_err = {score_err:.3e}");

        let got = {
            let mut v = vec![0.0f32; SEQ * HEADS * HD];
            d_out.copy_to(&mut v).unwrap();
            v
        };
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        println!("attention max_err = {max_err:.3e}");
        assert!(score_err < 1e-4, "parity scores: max_err={score_err}");
        assert!(max_err < 1e-4, "parity attention: max_err={max_err}");
    }

    #[test]
    fn bias_e_gather_parity_contro_cpu() {
        const SEQ: usize = 5;
        const NE: usize = 24;
        const VOCAB: usize = 11;
        let gpu = TestGpu::new();
        let stream = &gpu.0.stream;

        let x = lcg_input(SEQ * NE, 0xB1A5);
        let bias = lcg_input(NE, 0xB147);
        let table = lcg_input(VOCAB * NE, 0x7AB1E);
        let ids: Vec<u32> = (0..SEQ).map(|i| (i * 3 % VOCAB) as u32).collect();

        // ── add_bias ──
        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_bias = DeviceBuffer::from_slice(&bias).unwrap();
        launch_simple!(
            gpu.0.function("add_bias_f32"),
            stream,
            grid_for(SEQ * NE, 256),
            256,
            d_x.as_device_ptr(),
            d_bias.as_device_ptr(),
            SEQ as u32,
            NE as u32,
            1
        );
        let mut got_bias = vec![0.0f32; SEQ * NE];
        d_x.copy_to(&mut got_bias).unwrap();
        for s in 0..SEQ {
            for i in 0..NE {
                let expect = x[s * NE + i] + bias[i];
                assert!(
                    (got_bias[s * NE + i] - expect).abs() < 1e-6,
                    "bias[{s},{i}]"
                );
            }
        }

        // ── gather ──
        let d_table = DeviceBuffer::from_slice(&table).unwrap();
        let d_ids = DeviceBuffer::from_slice(&ids).unwrap();
        let mut d_out = DeviceBuffer::<f32>::zeroed(SEQ * NE).unwrap();
        launch_simple!(
            gpu.0.function("gather_rows_f32"),
            stream,
            grid_for(SEQ * NE, 256),
            256,
            d_table.as_device_ptr(),
            d_ids.as_device_ptr(),
            d_out.as_device_ptr(),
            SEQ as u32,
            NE as u32
        );
        let mut got_gather = vec![0.0f32; SEQ * NE];
        d_out.copy_to(&mut got_gather).unwrap();
        for s in 0..SEQ {
            let off = ids[s] as usize * NE;
            assert_eq!(&got_gather[s * NE..(s + 1) * NE], &table[off..off + NE]);
        }

        // ── copy_n con dst offsettato (append cache) ──
        let src = lcg_input(7, 0xC07A1);
        let mut d_dst = DeviceBuffer::<f32>::zeroed(16).unwrap();
        let d_src = DeviceBuffer::from_slice(&src).unwrap();
        launch_simple!(
            gpu.0.function("copy_n_f32"),
            stream,
            16,
            256,
            d_src.as_device_ptr(),
            ptr_off(d_dst.as_device_ptr(), 9),
            7
        );
        let mut got_copy = vec![0.0f32; 16];
        d_dst.copy_to(&mut got_copy).unwrap();
        assert_eq!(&got_copy[..9], &[0.0; 9]);
        assert_eq!(&got_copy[9..16], &src[..]);
        assert_eq!(got_copy[16 - 1], src[6]);
    }

    /// TEST DEFINITIVO F3b step 4: generazione greedy multi-token su GPU
    /// identica al CPU (che è parity llama.cpp da F1/F2).
    ///
    /// Attivazione (su GPU, modello presente):
    ///   S_CUDA_QWEN05B=/percorso/modello-q4km.gguf \
    ///     cargo test -p s-backend-cuda --features cuda --release \
    ///     -- --nocapture --ignored gpu_forward_generazione_identica_al_cpu
    #[test]
    #[ignore = "richiede modello reale + VRAM libera (~2.5GB): attivare con S_CUDA_QWEN05B"]
    fn gpu_forward_generazione_identica_al_cpu() {
        let Ok(path) = std::env::var("S_CUDA_QWEN05B") else {
            eprintln!("skip: S_CUDA_QWEN05B non impostata");
            return;
        };

        let cpu_model = Model::load(&path).unwrap();
        let tok_ids: Vec<u32> = vec![
            12050, 3852, 460, 773, 18920, 5652, 83435, 3950, 3524, 3375, 0, 3393, 220, 16, 17, 18,
            13,
        ];

        // riferimento CPU: greedy 12 token col path cached verificato
        let mut cpu_logits;
        let mut cpu_cache = cpu_model.new_cache(tok_ids.len() + 16);
        cpu_logits = cpu_model.forward_cached(&tok_ids, &mut cpu_cache);
        let mut cpu_tokens = Vec::new();
        for _ in 0..12 {
            let next = cpu_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            cpu_tokens.push(next);
            cpu_logits = cpu_model.forward_cached(&[next], &mut cpu_cache);
        }

        // GPU: stesso prompt, stesso loop
        let gpu_model = GpuQwen2::load(&cpu_model, tok_ids.len() + 16).unwrap();
        eprintln!("gpu: {}", gpu_model.describe().unwrap());
        let mut gpu_cache = gpu_model.new_cache(tok_ids.len() + 16);
        let mut gpu_logits = gpu_model.forward_cached(&tok_ids, &mut gpu_cache);
        let mut gpu_tokens = Vec::new();
        for _ in 0..12 {
            let next = gpu_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            gpu_tokens.push(next);
            gpu_logits = gpu_model.forward_cached(&[next], &mut gpu_cache);
        }

        assert_eq!(gpu_tokens, cpu_tokens, "generazione greedy divergente");
        eprintln!("✅ GPU == CPU su {} token greedy: {:?}", cpu_tokens.len(), cpu_tokens);
    }
}

