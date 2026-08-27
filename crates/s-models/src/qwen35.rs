//! Architettura qwen35 — ibrida Gated DeltaNet + Full Attention.
//!
//! F5: forward CPU naive f32 per Qwen3.5 (famiglia ibrida lineare+attention).
//! - 6 × (3× Gated DeltaNet → FFN → 1× Gated Attention → FFN) = 24 layer
//! - DeltaNet: recurrence O(1) per token, stato [d_state × d_state × n_heads]
//! - Full Attention: GQA con RoPE + Q/K norm
//!
//! Verifica F5: parity dei logits con llama.cpp sullo stesso prompt.

use crate::registry::Famiglia;
use crate::state::{LayerKv, LayerSeq, RecurrentState, SeqState};
use s_gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    // Full attention params (usati solo nei layer attention, non DeltaNet)
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,        // QKV internal dim (256 for Qwen3.5)
    pub attn_head_dim: usize,   // real RoPE head dim (256)
    pub rope_theta: f32,
    pub rope_n_rot: usize,      // RoPE dimension_count (64 for Qwen3.5 0.8B)
    pub rms_eps: f32,
    pub vocab_size: usize,
    // DeltaNet params
    pub ssm_d_state: usize,    // dimensione stato per testa (128)
    pub ssm_n_group: usize,    // numero teste QK (16)
    pub ssm_dt_rank: usize,    // numero teste V (16)
    pub ssm_d_inner: usize,    // dimensione interna (2048)
    pub ssm_d_conv: usize,     // kernel conv (4)
    pub full_attn_interval: usize, // intervallo attention (4)
}

impl Config {
    /// Config dal metadata GGUF (chiavi qwen35.*).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let get_u = |suffix: &str| -> Result<usize, String> {
            let key = format!("qwen35.{suffix}");
            gguf.metadata.iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v.as_u64().map(|v| v as usize))
                .ok_or(format!("metadata mancante: {key}"))
        };
        let get_f = |suffix: &str| -> Result<f32, String> {
            let key = format!("qwen35.{suffix}");
            gguf.metadata.iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v.as_f32())
                .ok_or(format!("metadata mancante: {key}"))
        };

        let n_embd = get_u("embedding_length")?;
        let n_heads = get_u("attention.head_count")?;
        let n_kv_heads = get_u("attention.head_count_kv")?;
        let ssm_n_group = get_u("ssm.group_count")?;
        let ssm_dt_rank = get_u("ssm.time_step_rank")?;
        let ssm_d_inner = get_u("ssm.inner_size")?;

        let hd = get_u("attention.key_length")?; // 256 for full attn
        let rope_n_rot = {
            let key = "qwen35.rope.dimension_count";
            gguf.metadata.iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v.as_u64().map(|v| v as usize))
                .unwrap_or(hd)
        };
        let block_count = get_u("block_count")?;
        let n_nextn = {
            let key = "qwen35.nextn_predict_layers";
            gguf.metadata.iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v.as_u64().map(|v| v as usize))
                .unwrap_or(0)
        };
        let n_layers = block_count.saturating_sub(n_nextn);
        Ok(Self {
            n_layers,
            n_embd,
            n_ff: get_u("feed_forward_length")?,
            n_heads,
            n_kv_heads,
            head_dim: hd,
            attn_head_dim: hd,
            rope_theta: get_f("rope.freq_base")?,
            rope_n_rot,
            rms_eps: get_f("attention.layer_norm_rms_epsilon")?,
            vocab_size: gguf.tensor("token_embd.weight")
                .map_err(|e| e.to_string())?
                .dims[1] as usize,
            ssm_d_state: get_u("ssm.state_size")?,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            ssm_d_conv: get_u("ssm.conv_kernel")?,
            full_attn_interval: get_u("full_attention_interval")?,
        })
    }

    /// Determina se un layer è DeltaNet (ricorrente) dato il suo indice.
    pub fn is_recurrent(&self, li: usize) -> bool {
        // Layer 0-based: full attention ogni full_attn_interval layer
        // (li+1) % interval != 0 → DeltaNet
        (li + 1) % self.full_attn_interval != 0
    }
}

/// Pesi di un layer DeltaNet.
struct DeltaNetWeights {
    attn_norm: Vec<f32>,
    qkv: Vec<f32>,           // [n_embd, key_dim*2 + value_dim]
    gate: Vec<f32>,           // [n_embd, value_dim] = z projection
    conv1d: Vec<f32>,         // [ssm_d_conv, conv_dim]
    dt_bias: Vec<f32>,        // [ssm_dt_rank] = [n_v_heads]
    ssm_a: Vec<f32>,          // [ssm_dt_rank] = [n_v_heads] (raw, in log space)
    alpha: Vec<f32>,          // [n_embd, n_v_heads]
    beta: Vec<f32>,           // [n_embd, n_v_heads]
    ssm_norm: Vec<f32>,       // [head_v_dim]
    ssm_out: Vec<f32>,        // [value_dim, n_embd]
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

/// Pesi di un layer Attention (GQA full, con Q/K norm).
struct AttentionWeights {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    #[allow(dead_code)] // usati solo se il GGUF indica QK norm con pesi !== None
    q_norm: Vec<f32>,
    #[allow(dead_code)]
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

enum LayerWeights {
    DeltaNet(DeltaNetWeights),
    Attention(AttentionWeights),
}

/// Modello Qwen3.5 caricato in memoria (pesi f32 dequantizzati).
pub struct Model {
    pub config: Config,
    tok_embd: Vec<f32>,
    output: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
}

fn dequant_tensor(gguf: &mut GgufFile, name: &str) -> Result<Vec<f32>, String> {
    gguf.tensor_data_f32(name).map_err(|e| format!("{name}: {e}"))
}

impl Model {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let mut gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
        let _famiglia = Famiglia::detect(&gguf)?;
        let config = Config::from_gguf(&gguf)?;

        eprintln!(
            "load qwen35: {} layer, {} embd, {} ffn, eps={}, theta={} | \
             DeltaNet: state={}, QK_heads={}, V_heads={}, conv_k={}, d_inner={} | \
             Attn: {}Q/{}KV, head_dim={}",
            config.n_layers, config.n_embd, config.n_ff, config.rms_eps, config.rope_theta,
            config.ssm_d_state, config.ssm_n_group, config.ssm_dt_rank,
            config.ssm_d_conv, config.ssm_d_inner,
            config.n_heads, config.n_kv_heads, config.head_dim,
        );

        let tok_embd = dequant_tensor(&mut gguf, "token_embd.weight")?;
        let output = match gguf.tensor("output.weight") {
            Ok(_) => dequant_tensor(&mut gguf, "output.weight")?,
            Err(_) => tok_embd.clone(), // tied embeddings (qwen3.5 0.8B)
        };
        let output_norm = match gguf.tensor("output_norm.weight") {
            Ok(_) => dequant_tensor(&mut gguf, "output_norm.weight")?,
            Err(e) => return Err(format!("output_norm.weight: {e}")),
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut l = |n: &str| -> Result<Vec<f32>, String> {
                dequant_tensor(&mut gguf, &format!("blk.{i}.{n}"))
            };

            if config.is_recurrent(i) {
                // DeltaNet layer
                layers.push(LayerWeights::DeltaNet(DeltaNetWeights {
                    attn_norm: l("attn_norm.weight")?,
                    qkv: l("attn_qkv.weight")?,
                    gate: l("attn_gate.weight")?,
                    conv1d: l("ssm_conv1d.weight")?,
                    dt_bias: l("ssm_dt.bias")?,
                    ssm_a: l("ssm_a")?,
                    alpha: l("ssm_alpha.weight")?,
                    beta: l("ssm_beta.weight")?,
                    ssm_norm: l("ssm_norm.weight")?,
                    ssm_out: l("ssm_out.weight")?,
                    post_norm: l("post_attention_norm.weight")?,
                    ffn_gate: l("ffn_gate.weight")?,
                    ffn_up: l("ffn_up.weight")?,
                    ffn_down: l("ffn_down.weight")?,
                }));
            } else {
                // Full attention layer
                layers.push(LayerWeights::Attention(AttentionWeights {
                    attn_norm: l("attn_norm.weight")?,
                    wq: l("attn_q.weight")?,
                    wk: l("attn_k.weight")?,
                    wv: l("attn_v.weight")?,
                    wo: l("attn_output.weight")?,
                    q_norm: l("attn_q_norm.weight")?,
                    k_norm: l("attn_k_norm.weight")?,
                    post_norm: l("post_attention_norm.weight")?,
                    ffn_gate: l("ffn_gate.weight")?,
                    ffn_up: l("ffn_up.weight")?,
                    ffn_down: l("ffn_down.weight")?,
                }));
            }
            if (i + 1) % 6 == 0 {
                eprintln!("load qwen35: {}/{} layer", i + 1, config.n_layers);
            }
        }
        Ok(Self { config, tok_embd, output, output_norm, layers })
    }

    /// Forward: token IDs → logits [vocab] (full-recompute, no cache).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.config;
        let seq = tokens.len();
        let ne = c.n_embd;

        // embedding lookup
        let mut x = vec![0.0f32; seq * ne];
        for (s, &t) in tokens.iter().enumerate() {
            let off = t as usize * ne;
            x[s * ne..(s + 1) * ne].copy_from_slice(&self.tok_embd[off..off + ne]);
        }

        // Buffer per conv state per layer (conv_k righe, la corrente + history)
        let key_dim = c.ssm_d_state * c.ssm_n_group;
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let mut conv_bufs: Vec<Vec<f32>> = vec![
            vec![0.0; c.ssm_d_conv * conv_dim];
            c.n_layers
        ];

        // Buffer per attention KV cache (per i layer attention)
        let kv_dim = c.n_kv_heads * c.head_dim;

        // Stato ricorrente DeltaNet: [d_state × d_state × n_v_heads] per layer
        let state_dim = c.ssm_d_state;
        let n_v_heads = c.ssm_dt_rank;
        let mut recurrent_states: Vec<Vec<f32>> = vec![
            vec![0.0; state_dim * state_dim * n_v_heads];
            c.n_layers
        ];

        let mut q_buf = vec![0.0f32; c.n_heads * c.head_dim]; // per attn layer
        let mut k_buf = vec![0.0f32; kv_dim]; // per attn layer
        let mut v_buf = vec![0.0f32; kv_dim]; // per attn layer

        let mut tmp_ffn1 = vec![0.0f32; c.n_ff];
        let mut tmp_ffn2 = vec![0.0f32; c.n_ff.max(ne)];

        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerWeights::DeltaNet(dn) => {
                    self.forward_delta_net_layer(
                        li, seq, &mut x, dn, c,
                        &mut conv_bufs[li], &mut recurrent_states[li],
                        &mut tmp_ffn1, &mut tmp_ffn2,
                    );
                }
                LayerWeights::Attention(at) => {
                    self.forward_attention_layer(
                        li, seq, &mut x, at, c,
                        &mut q_buf, &mut k_buf, &mut v_buf,
                        &mut tmp_ffn1, &mut tmp_ffn2,
                    );
                }
            }
        }

        // Final norm + output head
        let mut last_raw = vec![0.0f32; ne];
        last_raw.copy_from_slice(&x[(seq - 1) * ne..seq * ne]);
        let mut last_normed = vec![0.0f32; ne];
        rmsnorm_row(&last_raw, &self.output_norm, c.rms_eps, &mut last_normed);
        logits_from_hidden(&self.output, ne, &last_normed)
    }

    /// Crea uno SeqState dimensionato per `max_seq` posizioni.
    /// Layer ricorrenti → RecurrentState delta_net (S + conv_buf), layer attention → Kv.
    pub fn new_cache(&self, max_seq: usize) -> SeqState {
        let c = &self.config;
        let key_dim = c.ssm_d_state * c.ssm_n_group;
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            if c.is_recurrent(i) {
                let shape = [c.ssm_dt_rank, c.ssm_d_state, c.ssm_d_state];
                layers.push(LayerSeq::Recurrent(RecurrentState::delta_net(
                    shape, conv_dim, c.ssm_d_conv,
                )));
            } else {
                layers.push(LayerSeq::Kv(LayerKv::new(kv_dim, max_seq)));
            }
        }
        SeqState::from_layers(layers)
    }

    /// Forward cached: prefill (seq>1) o decode (seq==1) con stato persistente.
    /// Invariante: `forward(tokens) == new_cache + forward_cached(tokens)`.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut SeqState) -> Vec<f32> {
        assert_eq!(
            cache.n_layers(),
            self.config.n_layers,
            "cache n_layers {} != config {}",
            cache.n_layers(),
            self.config.n_layers
        );
        let c = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "forward_cached: tokens vuoto");
        let pos0 = cache.len;
        let ne = c.n_embd;

        // embedding lookup solo sui nuovi token
        let mut x = vec![0.0f32; seq * ne];
        for (s, &t) in tokens.iter().enumerate() {
            let off = t as usize * ne;
            x[s * ne..(s + 1) * ne].copy_from_slice(&self.tok_embd[off..off + ne]);
        }

        let mut tmp_ffn1 = vec![0.0f32; c.n_ff];
        let mut tmp_ffn2 = vec![0.0f32; c.n_ff.max(ne)];

        for (li, layer) in self.layers.iter().enumerate() {
            if c.is_recurrent(li) {
                // DeltaNet: usa S + conv_buf persistenti
                let rec = cache.layer_recurrent_mut(li);
                let (state, conv_buf) = rec.parts_mut();
                // forward_delta_net_layer si aspetta seq e x mut
                self.forward_delta_net_layer(
                    li, seq, &mut x,
                    match layer {
                        LayerWeights::DeltaNet(dn) => dn,
                        _ => panic!("layer {li} atteso DeltaNet"),
                    },
                    c, conv_buf, state, &mut tmp_ffn1, &mut tmp_ffn2,
                );
            } else {
                let at = match layer {
                    LayerWeights::Attention(a) => a,
                    _ => panic!("layer {li} atteso Attention"),
                };
                self.forward_attention_layer_cached(
                    li, seq, pos0, &mut x, at, c, cache, &mut tmp_ffn1, &mut tmp_ffn2,
                );
            }
        }

        cache.advance(seq);

        let mut last_raw = vec![0.0f32; ne];
        last_raw.copy_from_slice(&x[(seq - 1) * ne..seq * ne]);
        let mut last_normed = vec![0.0f32; ne];
        rmsnorm_row(&last_raw, &self.output_norm, c.rms_eps, &mut last_normed);
        logits_from_hidden(&self.output, ne, &last_normed)
    }

    pub fn debug_last_hidden(&self, tokens: &[u32]) -> Vec<f32> {
        self.debug_per_layer_hidden(tokens).last().cloned().unwrap()
    }

    pub fn debug_per_layer_hidden(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let c = &self.config;
        let seq = tokens.len();
        let ne = c.n_embd;
        let mut x = vec![0.0f32; seq * ne];
        for (s, &t) in tokens.iter().enumerate() {
            let off = t as usize * ne;
            x[s * ne..(s + 1) * ne].copy_from_slice(&self.tok_embd[off..off + ne]);
        }
        let key_dim = c.ssm_d_state * c.ssm_n_group;
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let mut conv_bufs: Vec<Vec<f32>> = vec![
            vec![0.0; c.ssm_d_conv * conv_dim];
            c.n_layers
        ];
        let state_dim = c.ssm_d_state;
        let n_v_heads = c.ssm_dt_rank;
        let mut recurrent_states: Vec<Vec<f32>> = vec![vec![0.0; state_dim * state_dim * n_v_heads]; c.n_layers];
        let mut q_buf = vec![0.0f32; c.n_heads * c.head_dim];
        let mut k_buf = vec![0.0f32; c.n_kv_heads * c.head_dim];
        let mut v_buf = vec![0.0f32; c.n_kv_heads * c.head_dim];
        let mut tmp_ffn1 = vec![0.0f32; c.n_ff];
        let mut tmp_ffn2 = vec![0.0f32; c.n_ff.max(ne)];
        let mut per_layer = Vec::new();
        // embedding is layer 0 in HF
        per_layer.push(x.clone());
        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerWeights::DeltaNet(dn) => {
                    self.forward_delta_net_layer(li, seq, &mut x, dn, c, &mut conv_bufs[li], &mut recurrent_states[li], &mut tmp_ffn1, &mut tmp_ffn2);
                }
                LayerWeights::Attention(at) => {
                    self.forward_attention_layer(li, seq, &mut x, at, c, &mut q_buf, &mut k_buf, &mut v_buf, &mut tmp_ffn1, &mut tmp_ffn2);
                }
            }
            per_layer.push(x.clone());
        }
        // final norm as last hidden (like HF layer 24)
        let mut last_raw = vec![0.0f32; ne];
        last_raw.copy_from_slice(&x[(seq - 1) * ne..seq * ne]);
        let mut last_normed = vec![0.0f32; ne];
        rmsnorm_row(&last_raw, &self.output_norm, c.rms_eps, &mut last_normed);
        // replace last entry with normed single token? For seq=1, per_layer last is before norm, we add normed as extra
        // To match HF's 25 hidden (embedding + 24 layers), we will replace last hidden with normed for comparison
        // Keep per_layer as is and add normed as separate
        per_layer.push(last_normed.clone());
        per_layer
    }
}

// ═══════════════════════════════════════════════════════════════
// DeltaNet layer forward (autoregressive, token-by-token)
// ═══════════════════════════════════════════════════════════════

impl Model {
    fn forward_delta_net_layer(
        &self,
        _li: usize,
        seq: usize,
        x: &mut [f32],
        dn: &DeltaNetWeights,
        c: &Config,
        conv_buf: &mut [f32],          // [(conv_k-1) × conv_dim]
        state: &mut [f32],             // [d_state × d_state × n_v_heads]
        tmp1: &mut [f32],
        tmp2: &mut [f32],
    ) {
        let ne = c.n_embd;
        let key_dim = c.ssm_d_state * c.ssm_n_group;   // 2048
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;  // 2048
        let conv_dim = key_dim * 2 + value_dim;          // 6144
        let head_v_dim = c.ssm_d_inner / c.ssm_dt_rank; // 128
        let n_v_heads = c.ssm_dt_rank;                   // 16
        let n_k_heads = c.ssm_n_group;                   // 16
        let d_state = c.ssm_d_state;                     // 128
        let conv_k = c.ssm_d_conv;                       // 4

        for s in 0..seq {
            let x_s = &x[s * ne..(s + 1) * ne];

            // 1. Input norm
            let mut normed = vec![0.0f32; ne];
            rmsnorm_row(x_s, &dn.attn_norm, c.rms_eps, &mut normed);

            // 2. QKV projection: [ne] @ [ne, 6144] = [6144]
            let mut qkv_mixed = vec![0.0f32; conv_dim];
            matmul_row_vec(&dn.qkv, ne, conv_dim, &normed, &mut qkv_mixed);

            // 3. Gate z: [ne] @ [ne, 2048] = [2048]
            let mut z = vec![0.0f32; value_dim];
            matmul_row_vec(&dn.gate, ne, value_dim, &normed, &mut z);

            // 4. Conv1D: shift conv_buf left, append qkv_mixed, compute conv output
            //    conv_buf has (conv_k-1) rows of conv_dim each
            //    Shift: move rows [1..] to [0..], write new row at end
            let row_size = conv_dim;
            for r in 0..(conv_k - 1) {
                let src = (r + 1) * row_size;
                let dst = r * row_size;
                conv_buf.copy_within(src..src + row_size, dst);
            }
            // Write new row
            let last_row = (conv_k - 1) * row_size;
            conv_buf[last_row..last_row + row_size].copy_from_slice(&qkv_mixed);

            // Conv output: depthwise causal conv (groups=conv_dim, kernel=4)
            // weight layout in GGUF/HF is [conv_dim, 1, conv_k] = [6144][4] = channel*4 + k
            // buf layout is [conv_k][conv_dim] = k*conv_dim + d
            let mut conv_out = vec![0.0f32; conv_dim];
            for d in 0..conv_dim {
                let mut sum = 0.0f32;
                for k_idx in 0..conv_k {
                    let w = dn.conv1d[d * conv_k + k_idx];
                    let b = conv_buf[k_idx * row_size + d];
                    sum += w * b;
                }
                conv_out[d] = sum;
            }

            // 5. SiLU activation on conv output
            for v in conv_out.iter_mut() {
                *v = silu(*v);
            }

            // 6. Split conv_out into Q, K, V
            //    Q: [d_state, n_k_heads] = [128, 16] — first key_dim elements per head
            //    K: [d_state, n_k_heads] = [128, 16] — next key_dim elements
            //    V: [head_v_dim, n_v_heads] = [128, 16] — next value_dim elements
            //    conv_out layout: [key_dim | key_dim | value_dim] = [2048 | 2048 | 2048]
            let q_k_flat = &conv_out[..key_dim];         // [2048] = [d_state * n_k_heads]
            let k_k_flat = &conv_out[key_dim..2 * key_dim]; // [2048]
            let v_flat = &conv_out[2 * key_dim..];         // [2048] = [head_v_dim * n_v_heads]

            // 7. L2 norm on Q and K (per head)
            //    q shape: [d_state, n_k_heads] → norm each [d_state] chunk
            //    k shape: [d_state, n_k_heads] → norm each [d_state] chunk
            let mut q_normed = vec![0.0f32; key_dim];
            let mut k_normed = vec![0.0f32; key_dim];
            for h in 0..n_k_heads {
                let h_start = h * d_state;
                let h_end = h_start + d_state;
                l2norm(&q_k_flat[h_start..h_end], &mut q_normed[h_start..h_end]);
                l2norm(&k_k_flat[h_start..h_end], &mut k_normed[h_start..h_end]);
            }

            // 8. Compute gate g = -exp(A_log) * softplus(alpha @ x + dt_bias)
            //    HF: g = -exp(A_log) * softplus(a + dt_bias), A_log = ssm_a
            //    alpha: [ne, n_v_heads] → alpha @ x = [n_v_heads]
            //    dt_bias: [n_v_heads]
            //    ssm_a: [n_v_heads] = A_log (log space)
            let mut g_vals = vec![0.0f32; n_v_heads];
            matmul_row_vec(&dn.alpha, ne, n_v_heads, &normed, &mut g_vals);
            for h in 0..n_v_heads {
                let a_log = dn.ssm_a[h];
                let sp = softplus(g_vals[h] + dn.dt_bias[h]);
                g_vals[h] = -a_log.exp() * sp;
            }

            // 9. Beta: [ne, n_v_heads] @ x → [n_v_heads], then sigmoid
            let mut beta_out = vec![0.0f32; n_v_heads];
            matmul_row_vec(&dn.beta, ne, n_v_heads, &normed, &mut beta_out);
            for v in beta_out.iter_mut() {
                *v = sigmoid(*v);
            }

            // 10. Scale Q by 1/sqrt(d_state)
            let scale = 1.0 / (d_state as f32).sqrt();
            for v in q_normed.iter_mut() {
                *v *= scale;
            }

            // 11. DeltaNet recurrence (autoregressive, single token)
            //     state: [d_state × d_state × n_v_heads]
            //     q_normed: [d_state × n_k_heads] — Q per testa
            //     k_normed: [d_state × n_k_heads] — K per testa
            //     v_flat: [head_v_dim × n_v_heads] — V per testa
            //     g: [n_v_heads] — gate (scalar per testa)
            //     b: [n_v_heads] — beta (scalar per testa)
            //
            //     Per testa h:
            //       S = S * exp(g[h])                          [d_state × d_state]
            //       sk = S @ k[h]                               [d_state]
            //       d = (v[h] - sk) * b[h]                      [d_state]
            //       S = S + k[h] ⊗ d                            [d_state × d_state]
            //       o[h] = (S @ q[h])                           [d_state]
            //
            //     KDA (g per-component): g has shape [d_state, n_v_heads]
            //     ma per Qwen3.5-0.8B è GDA (g scalar): g has shape [1, n_v_heads]

            let mut output_heads = vec![0.0f32; n_v_heads * head_v_dim];

            for h in 0..n_v_heads {
                let state_h = &mut state[h * d_state * d_state..(h + 1) * d_state * d_state];
                let k_h = &k_normed[h * d_state..(h + 1) * d_state];
                let v_h = &v_flat[h * head_v_dim..(h + 1) * head_v_dim];
                let q_h = &q_normed[h * d_state..(h + 1) * d_state];

                // Decay state: S *= exp(g)
                let g_exp = g_vals[h].exp();
                for s_i in state_h.iter_mut() {
                    *s_i *= g_exp;
                }

                // sk = S @ k
                let mut sk = vec![0.0f32; d_state];
                for j in 0..d_state {
                    let mut sum = 0.0f32;
                    for i in 0..d_state {
                        sum += state_h[j * d_state + i] * k_h[i];
                    }
                    sk[j] = sum;
                }

                // d = (v - sk) * b
                // V dimension = head_v_dim = d_state (128 == 128)
                // For n_k_heads != n_v_heads, need broadcast; here they're equal
                let mut d = vec![0.0f32; head_v_dim];
                for j in 0..head_v_dim {
                    d[j] = (v_h[j] - sk[j]) * beta_out[h];
                }

                // S += k ⊗ d (outer product)
                for j in 0..d_state {
                    for i in 0..d_state {
                        state_h[j * d_state + i] += k_h[i] * d[j];
                    }
                }

                // o = S @ q
                let mut o = vec![0.0f32; head_v_dim];
                for j in 0..head_v_dim {
                    let mut sum = 0.0f32;
                    for i in 0..d_state {
                        sum += state_h[j * d_state + i] * q_h[i];
                    }
                    o[j] = sum;
                }

                output_heads[h * head_v_dim..(h + 1) * head_v_dim].copy_from_slice(&o);
            }

            // 12. Gated norm: per-head rmsnorm(output_heads) * silu(z)
            //     ssm_norm: [head_v_dim] — lo applico a ogni testa
            let mut normed_out = vec![0.0f32; value_dim];
            for h in 0..n_v_heads {
                let h_start = h * head_v_dim;
                let h_end = h_start + head_v_dim;
                rmsnorm_row(
                    &output_heads[h_start..h_end],
                    &dn.ssm_norm,
                    c.rms_eps,
                    &mut normed_out[h_start..h_end],
                );
            }
            for i in 0..value_dim {
                normed_out[i] *= silu(z[i]);
            }

            // 13. Output projection: [value_dim] @ [value_dim, n_embd] = [n_embd]
            let mut attn_out = vec![0.0f32; ne];
            matmul_row_vec(&dn.ssm_out, value_dim, ne, &normed_out, &mut attn_out);

            // 14. Residual
            for i in 0..ne {
                x[s * ne + i] += attn_out[i];
            }

            // 15. Post-norm + FFN (SwiGLU)
            let x_s = &x[s * ne..(s + 1) * ne].to_vec();
            let mut post_normed = vec![0.0f32; ne];
            rmsnorm_row(&x_s, &dn.post_norm, c.rms_eps, &mut post_normed);

            // gate = silu(W_gate @ normed) * (W_up @ normed)
            let gate = &mut tmp1[..c.n_ff];
            matmul_row_vec(&dn.ffn_gate, ne, c.n_ff, &post_normed, gate);
            let up = &mut tmp2[..c.n_ff];
            matmul_row_vec(&dn.ffn_up, ne, c.n_ff, &post_normed, up);
            for i in 0..c.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }

            let down = &mut tmp2[..ne];
            matmul_row_vec(&dn.ffn_down, c.n_ff, ne, gate, down);
            for i in 0..ne {
                x[s * ne + i] += down[i];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Full Attention layer forward (GQA + RoPE + Q/K norm)
// ═══════════════════════════════════════════════════════════════

impl Model {
    fn forward_attention_layer(
        &self,
        _li: usize,
        seq: usize,
        x: &mut [f32],
        at: &AttentionWeights,
        c: &Config,
        _q_buf: &mut [f32],
        _k_buf: &mut [f32],
        _v_buf: &mut [f32],
        tmp1: &mut [f32],
        tmp2: &mut [f32],
    ) {
        let ne = c.n_embd;
        let hd = c.head_dim;
        let n_heads = c.n_heads;
        let n_kv_heads = c.n_kv_heads;
        let q_dim = n_heads * hd;          // 2048
        let q_full_dim = q_dim * 2;        // 4096 (Q + gate)
        let kv_dim = n_kv_heads * hd;      // 512
        let n_rot = c.rope_n_rot;          // 64

        // Temporary buffers
        let mut normed = vec![0.0f32; seq * ne];
        let mut q_all = vec![0.0f32; seq * q_dim];
        let mut gate_all = vec![0.0f32; seq * q_dim];
        let mut k_all = vec![0.0f32; seq * kv_dim];
        let mut v_all = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];

        for s in 0..seq {
            let x_s = &x[s * ne..(s + 1) * ne];
            rmsnorm_row(x_s, &at.attn_norm, c.rms_eps, &mut normed[s * ne..(s + 1) * ne]);
            let x_normed = &normed[s * ne..(s + 1) * ne];

            // Q full projection: [ne] -> [q_full_dim] then split
            let mut q_full = vec![0.0f32; q_full_dim];
            matmul_row_vec(&at.wq, ne, q_full_dim, x_normed, &mut q_full);
            let q_off = s * q_dim;
            let g_off = s * q_dim;
            let kv_off = s * kv_dim;

            // Split Q and gate per-head (q_proj is per-head interleaved: head 0 q 256, gate 256, head1 q, gate...)
            for h in 0..n_heads {
                let src = h * 512;
                let dst = h * hd;
                q_all[q_off + dst..q_off + dst + hd].copy_from_slice(&q_full[src..src + hd]);
                gate_all[g_off + dst..g_off + dst + hd].copy_from_slice(&q_full[src + hd..src + 2 * hd]);
            }

            // Q RMSNorm per head (weight broadcast [hd])
            for h in 0..n_heads {
                let h_off = q_off + h * hd;
                let mut tmp = vec![0.0f32; hd];
                rmsnorm_row(&q_all[h_off..h_off + hd], &at.q_norm, c.rms_eps, &mut tmp);
                q_all[h_off..h_off + hd].copy_from_slice(&tmp);
            }

            // K / V projection
            matmul_row_vec(&at.wk, ne, kv_dim, x_normed, &mut k_all[kv_off..kv_off + kv_dim]);
            matmul_row_vec(&at.wv, ne, kv_dim, x_normed, &mut v_all[kv_off..kv_off + kv_dim]);

            // K RMSNorm per head
            for h in 0..n_kv_heads {
                let h_off = kv_off + h * hd;
                let mut tmp = vec![0.0f32; hd];
                rmsnorm_row(&k_all[h_off..h_off + hd], &at.k_norm, c.rms_eps, &mut tmp);
                k_all[h_off..h_off + hd].copy_from_slice(&tmp);
            }

            // RoPE (NEOX half-split) — only first n_rot dims per head
            rope_mrope(&mut q_all[q_off..q_off + q_dim], n_heads, hd, c.rope_theta, s, n_rot);
            rope_mrope(&mut k_all[kv_off..kv_off + kv_dim], n_kv_heads, hd, c.rope_theta, s, n_rot);
        }

        // Causal GQA attention per token
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for s in 0..seq {
            for h in 0..n_heads {
                let kvh = h / heads_per_kv;
                let mut scores = vec![0.0f32; s + 1];
                for t in 0..=s {
                    let qh = &q_all[s * q_dim + h * hd..s * q_dim + (h + 1) * hd];
                    let kh = &k_all[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qh[d] * kh[d];
                    }
                    scores[t] = dot * scale;
                }
                softmax_row_f32(&mut scores);
                let o_off = s * q_dim + h * hd;
                for d in 0..hd {
                    attn_out[o_off + d] = 0.0;
                }
                for t in 0..=s {
                    let w = scores[t];
                    let vh = &v_all[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd {
                        attn_out[o_off + d] += w * vh[d];
                    }
                }
            }
        }

        // Gating: attn_out *= sigmoid(gate)
        for s in 0..seq {
            let base = s * q_dim;
            for i in 0..q_dim {
                attn_out[base + i] *= sigmoid(gate_all[base + i]);
            }
        }

        // Output projection + residual (wo: [q_dim, ne] -> [ne])
        for s in 0..seq {
            let mut proj = vec![0.0f32; ne];
            matmul_row_vec(&at.wo, q_dim, ne, &attn_out[s * q_dim..(s + 1) * q_dim], &mut proj);
            for i in 0..ne {
                x[s * ne + i] += proj[i];
            }
        }

        // Post-norm + FFN
        for s in 0..seq {
            let x_s = x[s * ne..(s + 1) * ne].to_vec();
            let mut post_normed = vec![0.0f32; ne];
            rmsnorm_row(&x_s, &at.post_norm, c.rms_eps, &mut post_normed);

            let gate = &mut tmp1[..c.n_ff];
            matmul_row_vec(&at.ffn_gate, ne, c.n_ff, &post_normed, gate);
            let up = &mut tmp2[..c.n_ff];
            matmul_row_vec(&at.ffn_up, ne, c.n_ff, &post_normed, up);
            for i in 0..c.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            let down = &mut tmp2[..ne];
            matmul_row_vec(&at.ffn_down, c.n_ff, ne, gate, down);
            for i in 0..ne {
                x[s * ne + i] += down[i];
            }
        }
    }

    /// Attention cached — F6: stesso calcolo di `forward_attention_layer` ma incrementale.
    /// `pos0 = cache.len` prima della scrittura; scrive K/V rope'd in cache e le rilegge per attention.
    fn forward_attention_layer_cached(
        &self,
        li: usize,
        seq: usize,
        pos0: usize,
        x: &mut [f32],
        at: &AttentionWeights,
        c: &Config,
        cache: &mut SeqState,
        tmp1: &mut [f32],
        tmp2: &mut [f32],
    ) {
        let ne = c.n_embd;
        let hd = c.head_dim;
        let n_heads = c.n_heads;
        let n_kv_heads = c.n_kv_heads;
        let q_dim = n_heads * hd;
        let q_full_dim = q_dim * 2;
        let kv_dim = n_kv_heads * hd;
        let n_rot = c.rope_n_rot;

        let mut normed = vec![0.0f32; seq * ne];
        let mut q_all = vec![0.0f32; seq * q_dim];
        let mut gate_all = vec![0.0f32; seq * q_dim];
        let mut k_new = vec![0.0f32; seq * kv_dim];
        let mut v_new = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];

        for s in 0..seq {
            let x_s = &x[s * ne..(s + 1) * ne];
            rmsnorm_row(x_s, &at.attn_norm, c.rms_eps, &mut normed[s * ne..(s + 1) * ne]);
            let x_normed = &normed[s * ne..(s + 1) * ne];

            let mut q_full = vec![0.0f32; q_full_dim];
            matmul_row_vec(&at.wq, ne, q_full_dim, x_normed, &mut q_full);
            let q_off = s * q_dim;
            let g_off = s * q_dim;
            let kv_off = s * kv_dim;
            for h in 0..n_heads {
                let src = h * 512;
                let dst = h * hd;
                q_all[q_off + dst..q_off + dst + hd].copy_from_slice(&q_full[src..src + hd]);
                gate_all[g_off + dst..g_off + dst + hd].copy_from_slice(&q_full[src + hd..src + 2 * hd]);
            }
            for h in 0..n_heads {
                let h_off = q_off + h * hd;
                let mut tmp = vec![0.0f32; hd];
                rmsnorm_row(&q_all[h_off..h_off + hd], &at.q_norm, c.rms_eps, &mut tmp);
                q_all[h_off..h_off + hd].copy_from_slice(&tmp);
            }
            matmul_row_vec(&at.wk, ne, kv_dim, x_normed, &mut k_new[kv_off..kv_off + kv_dim]);
            matmul_row_vec(&at.wv, ne, kv_dim, x_normed, &mut v_new[kv_off..kv_off + kv_dim]);
            for h in 0..n_kv_heads {
                let h_off = kv_off + h * hd;
                let mut tmp = vec![0.0f32; hd];
                rmsnorm_row(&k_new[h_off..h_off + hd], &at.k_norm, c.rms_eps, &mut tmp);
                k_new[h_off..h_off + hd].copy_from_slice(&tmp);
            }
            rope_mrope(&mut q_all[q_off..q_off + q_dim], n_heads, hd, c.rope_theta, pos0 + s, n_rot);
            rope_mrope(&mut k_new[kv_off..kv_off + kv_dim], n_kv_heads, hd, c.rope_theta, pos0 + s, n_rot);
        }

        // scrivi K/V nuovi in cache (layer_k_mut già offsettato a pos0)
        cache.layer_k_mut(li)[..seq * kv_dim].copy_from_slice(&k_new);
        cache.layer_v_mut(li)[..seq * kv_dim].copy_from_slice(&v_new);

        let total = pos0 + seq;
        // attenzione: k_all/v_all per il prefisso sono in cache, non più in k_new
        let k_cache = cache.layer_k_upto(li, total);
        let v_cache = cache.layer_v_upto(li, total);
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for s in 0..seq {
            for h in 0..n_heads {
                let kvh = h / heads_per_kv;
                let abs_s = pos0 + s;
                let mut scores = vec![0.0f32; abs_s + 1];
                for t in 0..=abs_s {
                    let qh = &q_all[s * q_dim + h * hd..s * q_dim + (h + 1) * hd];
                    let kh = &k_cache[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qh[d] * kh[d];
                    }
                    scores[t] = dot * scale;
                }
                softmax_row_f32(&mut scores);
                let o_off = s * q_dim + h * hd;
                for d in 0..hd {
                    attn_out[o_off + d] = 0.0;
                }
                for t in 0..=abs_s {
                    let w = scores[t];
                    let vh = &v_cache[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd {
                        attn_out[o_off + d] += w * vh[d];
                    }
                }
            }
        }

        for s in 0..seq {
            let base = s * q_dim;
            for i in 0..q_dim {
                attn_out[base + i] *= sigmoid(gate_all[base + i]);
            }
        }
        for s in 0..seq {
            let mut proj = vec![0.0f32; ne];
            matmul_row_vec(&at.wo, q_dim, ne, &attn_out[s * q_dim..(s + 1) * q_dim], &mut proj);
            for i in 0..ne {
                x[s * ne + i] += proj[i];
            }
        }
        for s in 0..seq {
            let x_s = x[s * ne..(s + 1) * ne].to_vec();
            let mut post_normed = vec![0.0f32; ne];
            rmsnorm_row(&x_s, &at.post_norm, c.rms_eps, &mut post_normed);
            let gate = &mut tmp1[..c.n_ff];
            matmul_row_vec(&at.ffn_gate, ne, c.n_ff, &post_normed, gate);
            let up = &mut tmp2[..c.n_ff];
            matmul_row_vec(&at.ffn_up, ne, c.n_ff, &post_normed, up);
            for i in 0..c.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            let down = &mut tmp2[..ne];
            matmul_row_vec(&at.ffn_down, c.n_ff, ne, gate, down);
            for i in 0..ne {
                x[s * ne + i] += down[i];
            }
        }
    }
    /// Debug: esegue un singolo layer attention (indice li) su hidden isolato (seq=1) per parity vs HF.
    pub fn debug_single_layer(&self, layer_idx: usize, hidden_in: &[f32]) -> Vec<f32> {
        let c = &self.config;
        let ne = c.n_embd;
        assert_eq!(hidden_in.len(), ne);
        let mut x = hidden_in.to_vec();
        let mut tmp1 = vec![0.0f32; c.n_ff];
        let mut tmp2 = vec![0.0f32; c.n_ff.max(ne)];
        let layer = &self.layers[layer_idx];
        // conv/recurrent state not needed for attention layers; create dummy
        let key_dim = c.ssm_d_state * c.ssm_n_group;
        let value_dim = c.ssm_d_state * c.ssm_dt_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let mut dummy_conv = vec![0.0f32; c.ssm_d_conv * conv_dim];
        let mut dummy_state = vec![0.0f32; c.ssm_d_state * c.ssm_d_state * c.ssm_dt_rank];
        let mut q_buf = vec![0.0f32; c.n_heads * c.head_dim];
        let mut k_buf = vec![0.0f32; c.n_kv_heads * c.head_dim];
        let mut v_buf = vec![0.0f32; c.n_kv_heads * c.head_dim];
        // x is currently single token, need to wrap as seq=1 slice
        let mut x_seq = x.clone(); // seq*ne with seq=1
        match layer {
            LayerWeights::DeltaNet(dn) => {
                self.forward_delta_net_layer(layer_idx, 1, &mut x_seq, dn, c, &mut dummy_conv, &mut dummy_state, &mut tmp1, &mut tmp2);
            }
            LayerWeights::Attention(at) => {
                self.forward_attention_layer(layer_idx, 1, &mut x_seq, at, c, &mut q_buf, &mut k_buf, &mut v_buf, &mut tmp1, &mut tmp2);
            }
        }
        x_seq
    }
}

pub struct Qwen35DeltaView<'a> {
    pub attn_norm: &'a [f32],
    pub qkv: &'a [f32],
    pub gate: &'a [f32],
    pub conv1d: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: &'a [f32],
    pub post_norm: &'a [f32],
    pub ffn_gate: &'a [f32],
    pub ffn_up: &'a [f32],
    pub ffn_down: &'a [f32],
}

pub struct Qwen35AttentionView<'a> {
    pub attn_norm: &'a [f32],
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub ffn_gate: &'a [f32],
    pub ffn_up: &'a [f32],
    pub ffn_down: &'a [f32],
}

pub enum Qwen35LayerView<'a> {
    DeltaNet(Qwen35DeltaView<'a>),
    Attention(Qwen35AttentionView<'a>),
}

pub struct Qwen35WeightViews<'a> {
    pub tok_embd: &'a [f32],
    pub output: &'a [f32],
    pub output_norm: &'a [f32],
    pub layers: Vec<Qwen35LayerView<'a>>,
}

impl Model {
    pub fn weight_views_for_gpu(&self) -> Qwen35WeightViews<'_> {
        Qwen35WeightViews {
            tok_embd: &self.tok_embd,
            output: &self.output,
            output_norm: &self.output_norm,
            layers: self.layers.iter().map(|l| match l {
                LayerWeights::DeltaNet(dn) => Qwen35LayerView::DeltaNet(Qwen35DeltaView {
                    attn_norm: &dn.attn_norm,
                    qkv: &dn.qkv,
                    gate: &dn.gate,
                    conv1d: &dn.conv1d,
                    dt_bias: &dn.dt_bias,
                    ssm_a: &dn.ssm_a,
                    alpha: &dn.alpha,
                    beta: &dn.beta,
                    ssm_norm: &dn.ssm_norm,
                    ssm_out: &dn.ssm_out,
                    post_norm: &dn.post_norm,
                    ffn_gate: &dn.ffn_gate,
                    ffn_up: &dn.ffn_up,
                    ffn_down: &dn.ffn_down,
                }),
                LayerWeights::Attention(at) => Qwen35LayerView::Attention(Qwen35AttentionView {
                    attn_norm: &at.attn_norm,
                    wq: &at.wq,
                    wk: &at.wk,
                    wv: &at.wv,
                    wo: &at.wo,
                    q_norm: &at.q_norm,
                    k_norm: &at.k_norm,
                    post_norm: &at.post_norm,
                    ffn_gate: &at.ffn_gate,
                    ffn_up: &at.ffn_up,
                    ffn_down: &at.ffn_down,
                }),
            }).collect(),
        }
    }
}

// ── Device impl — F6: Qwen35 come backend per generate.rs / s-server ──
impl crate::device::Device for Model {
    type Cache = SeqState;
    type Config = Config;

    fn config(&self) -> &Config {
        &self.config
    }

    fn new_cache(&self, max_seq: usize) -> SeqState {
        self.new_cache(max_seq)
    }

    fn forward_cached(&self, tokens: &[u32], cache: &mut SeqState) -> Vec<f32> {
        self.forward_cached(tokens, cache)
    }
}

// ═══════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════

/// RMSNorm: y = x * w / sqrt(mean(x²) + eps)
fn rmsnorm_row(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(weight.len(), n);
    let mut ss = 0.0f64;
    for &v in input {
        ss += v as f64 * v as f64;
    }
    let inv = 1.0 / ((ss / n as f64 + eps as f64) as f32).sqrt();
    for i in 0..n {
        output[i] = input[i] * inv * weight[i];
    }
}

/// L2 normalization: y = x / ||x||₂  (eps 1e-6 come HF/FLA)
fn l2norm(input: &[f32], output: &mut [f32]) {
    let mut ss = 0.0f64;
    for &v in input {
        ss += v as f64 * v as f64;
    }
    let inv = 1.0 / ((ss as f32) + 1e-6).sqrt();
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v * inv;
    }
}

/// L2 normalization in-place: x = x / ||x||₂
fn l2norm_inplace(data: &mut [f32]) {
    let mut ss = 0.0f64;
    for &v in data.iter() {
        ss += v as f64 * v as f64;
    }
    let inv = 1.0 / (ss as f32).sqrt();
    for v in data.iter_mut() {
        *v *= inv;
    }
}

/// Matmul row: output[out_dim] = input[in_dim] @ weight.T[out_dim, in_dim]
/// weight layout: [in_dim, out_dim] (row-major)
fn matmul_row_vec(weight: &[f32], in_dim: usize, out_dim: usize, input: &[f32], output: &mut [f32]) {
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        let w_row = o * in_dim;
        for i in 0..in_dim {
            sum += input[i] * weight[w_row + i];
        }
        output[o] = sum;
    }
}

fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// RoPE stile NEOX (half-split, coppie (i, i+hd/2))
fn rope_neox(data: &mut [f32], n_heads: usize, hd: usize, theta: f32, pos: usize) {
    let half = hd / 2;
    for h in 0..n_heads {
        let base = h * hd;
        for i in 0..half {
            let freq = theta.powf(-2.0 * i as f32 / hd as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let a = data[base + i];
            let b = data[base + i + half];
            data[base + i] = a * cos - b * sin;
            data[base + i + half] = a * sin + b * cos;
        }
    }
}

/// RoPE MRoPE parziale: ruota solo i primi n_rot dims per testa
fn rope_mrope(data: &mut [f32], n_heads: usize, hd: usize, theta: f32, pos: usize, n_rot: usize) {
    let half = hd / 2;
    let half_rot = n_rot / 2;
    for h in 0..n_heads {
        let base = h * hd;
        for i in 0..half_rot {
            let freq = theta.powf(-2.0 * i as f32 / hd as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let a = data[base + i];
            let b = data[base + i + half];
            data[base + i] = a * cos - b * sin;
            data[base + i + half] = a * sin + b * cos;
        }
    }
}

/// Softmax su una riga
fn softmax_row_f32(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Logits: last_hidden @ output^T
fn logits_from_hidden(output: &[f32], n_embd: usize, last_hidden: &[f32]) -> Vec<f32> {
    let vocab = output.len() / n_embd;
    let mut logits = vec![0.0f32; vocab];
    for o in 0..vocab {
        let w_off = o * n_embd;
        let mut sum = 0.0f32;
        for d in 0..n_embd {
            sum += output[w_off + d] * last_hidden[d];
        }
        logits[o] = sum;
    }
    logits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_forward_qwen35_08b() {
        let path = "/tmp/opencode/Qwen3.5-0.8B-Q4_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} non trovato");
            return;
        }
        let model = Model::load(path).expect("load");
        assert_eq!(model.config.n_layers, 24);
        assert_eq!(model.config.n_embd, 1024);
        assert_eq!(model.config.ssm_d_state, 128);
        assert_eq!(model.config.ssm_n_group, 16);
        assert_eq!(model.config.ssm_dt_rank, 16);
        assert_eq!(model.config.full_attn_interval, 4);

        // Forward su 4 token dummy (verifica che non panic)
        let tokens = vec![1u32, 2, 3, 4];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), model.config.vocab_size,
            "vocab_size atteso {}", model.config.vocab_size);
        eprintln!("forward ok: logits[0..5] = {:?}", &logits[..5.min(logits.len())]);
    }

    #[test]
    fn cache_parity_vs_full_recompute() {
        let path = "/tmp/opencode/Qwen3.5-0.8B-F16.gguf";
        let path_q4 = "/tmp/opencode/Qwen3.5-0.8B-Q4_0.gguf";
        let path = if std::path::Path::new(path).exists() { path } else { path_q4 };
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} non trovato");
            return;
        }
        let model = Model::load(path).expect("load");
        // prompt di test: Hello [9419] + 3 token extra
        let prompt_full = vec![9419u32, 11, 4858, 0];
        // golden full-recompute
        let logits_full = model.forward(&prompt_full);
        // cached: prefill 2 + decode 1 + decode 1
        let mut cache = model.new_cache(64);
        let l1 = model.forward_cached(&prompt_full[..2], &mut cache);
        let _ = l1; // logits di [9419,11] non confrontati (prefill intermedio)
        let l2 = model.forward_cached(&prompt_full[2..3], &mut cache);
        let logits_cached_step = model.forward_cached(&prompt_full[3..4], &mut cache);
        // ora confronta cached step finale vs full
        let maxdiff = logits_full.iter().zip(logits_cached_step.iter())
            .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let mean = logits_full.iter().zip(logits_cached_step.iter())
            .map(|(a,b)| (a-b).abs()).sum::<f32>() / logits_full.len() as f32;
        eprintln!("cache parity: maxdiff {maxdiff:e} mean {mean:e}");
        assert!(maxdiff < 1e-4, "maxdiff {maxdiff} troppo alto (atteso <1e-4)");
        // anche prefill unico vs full deve coincidere
        let mut cache2 = model.new_cache(64);
        let logits_prefill = model.forward_cached(&prompt_full, &mut cache2);
        let maxdiff2 = logits_full.iter().zip(logits_prefill.iter())
            .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        eprintln!("prefill parity: maxdiff {maxdiff2:e}");
        assert!(maxdiff2 < 1e-4, "prefill maxdiff {maxdiff2}");

        // prefill lungo (8 token) vs decode 1+1
        let long = vec![9419u32, 11, 4858, 0, 13, 1017, 13, 16];
        let full_long = model.forward(&long);
        let mut c3 = model.new_cache(64);
        let _ = model.forward_cached(&long[..4], &mut c3);
        let cached_long = model.forward_cached(&long[4..], &mut c3);
        // attenzione: forward_cached(&long[4..]) con seq=4 non è decode 1-per-1 ma chunk
        // confronto: dovrebbe coincidere con full su stesso prefisso totale
        let mut c4 = model.new_cache(64);
        let full_cached_long = model.forward_cached(&long, &mut c4);
        let md = full_long.iter().zip(full_cached_long.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
        eprintln!("long prefill chunk maxdiff {md:e} cached_chunk vs full_cached { }", {
            let d = cached_long.iter().zip(full_cached_long.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
            d
        });
        assert!(md < 1e-4);
    }

    #[test]
    fn generate_via_device_parity() {
        let path = "/tmp/opencode/Qwen3.5-0.8B-F16.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("skip"); return; }
        let model = Model::load(path).unwrap();
        let prompt = vec![9419u32, 11, 4858];
        // manual loop 6 token
        let mut cache = model.new_cache(64);
        let mut logits = model.forward_cached(&prompt, &mut cache);
        let mut manual = Vec::new();
        for _ in 0..6 {
            let next = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            manual.push(next);
            logits = model.forward_cached(&[next], &mut cache);
        }
        // via generate trait
        use crate::generate::{GenerateParams, generate_tokens};
        use crate::device::Device as _;
        let params = GenerateParams { max_tokens: 6, temperature: 0.0, ..Default::default() };
        let eos: Vec<u32> = vec![];
        let out = generate_tokens(&model, &prompt, &params, &eos, &|ids| ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), &mut |_| true, &mut || None);
        assert_eq!(out.ids, manual, "generate via Device diverge da manual cached loop");
        eprintln!("generate parity 6 tok: {:?}", out.ids);
    }
}
