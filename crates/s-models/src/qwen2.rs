//! Architettura qwen2 (famiglia llama con bias QKV): forward CPU naive f32.
//!
//! Design F1: semplicità e correttezza prima della velocità.
//! - Pesi dequantizzati in f32 al load (0.5B → ~2GB RAM, accettabile).
//! - Attenzione causale full-recompute (niente KV cache: la porta s-kv in F2).
//! - RoPE stile NEOX (half-split, coppie (i, i+hd/2)) — verificato empiricamente.
//!
//! Verifica F1: generazione greedy token-identica a llama.cpp.

use s_gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab_size: usize,
}

impl Config {
    /// Config dal metadata GGUF (chiavi qwen2.* o llama.*).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let get_u = |suffix: &str| -> Result<usize, String> {
            for prefix in ["qwen2", "llama"] {
                let key = format!("{prefix}.{suffix}");
                if let Some((_, v)) = gguf.metadata.iter().find(|(k, _)| *k == key) {
                    return v.as_u64().map(|v| v as usize).ok_or(format!("{key}: tipo inatteso"));
                }
            }
            Err(format!("metadata mancante: {suffix}"))
        };
        let get_f = |suffix: &str| -> Result<f32, String> {
            for prefix in ["qwen2", "llama"] {
                let key = format!("{prefix}.{suffix}");
                if let Some((_, v)) = gguf.metadata.iter().find(|(k, _)| *k == key) {
                    return v.as_f32().ok_or(format!("{key}: tipo inatteso"));
                }
            }
            Err(format!("metadata mancante: {suffix}"))
        };
        let n_embd = get_u("embedding_length")?;
        let n_heads = get_u("attention.head_count")?;
        let n_kv_heads = get_u("attention.head_count_kv")?;
        Ok(Self {
            n_layers: get_u("block_count")?,
            n_embd,
            n_ff: get_u("feed_forward_length")?,
            n_heads,
            n_kv_heads,
            head_dim: n_embd / n_heads,
            rope_theta: get_f("rope.freq_base")?,
            rms_eps: get_f("attention.layer_norm_rms_epsilon")?,
            vocab_size: gguf.tensor("token_embd.weight").map_err(|e| e.to_string())?.dims[1] as usize,
        })
    }
}

/// Modello caricato in memoria (pesi f32 dequantizzati).
pub struct Model {
    pub config: Config,
    /// embedding [vocab × n_embd]
    tok_embd: Vec<f32>,
    /// output head [vocab × n_embd] (può condividere con tok_embd se assente)
    output: Vec<f32>,
    /// pesi del final norm (output_norm.weight)
    output_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    bq: Vec<f32>,
    wk: Vec<f32>,
    bk: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

fn dequant_tensor(gguf: &mut GgufFile, name: &str) -> Result<Vec<f32>, String> {
    gguf.tensor_data_f32(name).map_err(|e| format!("{name}: {e}"))
}

impl Model {
    /// Carica il modello dequantizzando tutti i pesi in f32.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let mut gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = Config::from_gguf(&gguf)?;
        eprintln!(
            "load: {} layer, {} embd, {} heads ({} kv), {} ffn, eps={}, theta={}",
            config.n_layers, config.n_embd, config.n_heads, config.n_kv_heads, config.n_ff,
            config.rms_eps, config.rope_theta
        );

        let tok_embd = dequant_tensor(&mut gguf, "token_embd.weight")?;
        // output.weight separato (qwen2.5 non è tied); fallback: tied embeddings
        let output = match gguf.tensor("output.weight") {
            Ok(_) => dequant_tensor(&mut gguf, "output.weight")?,
            Err(_) => tok_embd.clone(),
        };

        let output_norm = match gguf.tensor("output_norm.weight") {
            Ok(_) => dequant_tensor(&mut gguf, "output_norm.weight")?,
            Err(e) => return Err(format!("output_norm.weight: {e}")),
        };
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut l = |n: &str| -> Result<Vec<f32>, String> { dequant_tensor(&mut gguf, &format!("blk.{i}.{n}")) };
            layers.push(LayerWeights {
                attn_norm: l("attn_norm.weight")?,
                wq: l("attn_q.weight")?,
                bq: l("attn_q.bias")?,
                wk: l("attn_k.weight")?,
                bk: l("attn_k.bias")?,
                wv: l("attn_v.weight")?,
                bv: l("attn_v.bias")?,
                wo: l("attn_output.weight")?,
                ffn_norm: l("ffn_norm.weight")?,
                w_gate: l("ffn_gate.weight")?,
                w_up: l("ffn_up.weight")?,
                w_down: l("ffn_down.weight")?,
            });
            if (i + 1) % 8 == 0 {
                eprintln!("load: {}/{} layer", i + 1, config.n_layers);
            }
        }
        Ok(Self { config, tok_embd, output, output_norm, layers })
    }

    /// Forward con traccia diagnostica su stderr (per il debug parity).
    pub fn forward_traced(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.config;
        let seq = tokens.len();
        let (ne, hd) = (c.n_embd, c.head_dim);
        let kv_dim = c.n_kv_heads * hd;

        fn stats(tag: &str, x: &[f32]) {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            let mut nan = 0usize;
            let mut inf = 0usize;
            let mut huge = 0usize;
            for &v in x {
                if v.is_nan() { nan += 1; continue; }
                if v.is_infinite() { inf += 1; continue; }
                if v.abs() > 1e6 { huge += 1; }
                min = min.min(v);
                max = max.max(v);
                sum += (v as f64).abs();
            }
            eprintln!(
                "{:<28} min={min:+.4e} max={max:+.4e} |m|={:.4e} nan={nan} inf={inf} >1e6={huge}",
                tag,
                (sum / x.len() as f64) as f32
            );
        }

        // embedding lookup
        let mut x = vec![0.0f32; seq * ne];
        for (s, &t) in tokens.iter().enumerate() {
            let off = t as usize * ne;
            x[s * ne..(s + 1) * ne].copy_from_slice(&self.tok_embd[off..off + ne]);
        }
        stats("embedding", &x);
        if let Some(&t0) = tokens.first() {
            let off = t0 as usize * ne;
            eprintln!("emb[{}][0..6]={:?}", t0, &self.tok_embd[off..off + 6]);
        }

        // statistiche pesi per layer 0
        {
            let l = &self.layers[0];
            for (name, w) in [
                ("attn_norm", &l.attn_norm), ("wq", &l.wq), ("bq", &l.bq),
                ("wv", &l.wv), ("wo", &l.wo), ("w_gate", &l.w_gate),
                ("w_up", &l.w_up), ("w_down", &l.w_down),
            ] {
                stats(&format!("pesi blk0.{name}"), w);
            }
        }

        let mut normed = vec![0.0f32; seq * ne];
        let mut q = vec![0.0f32; seq * ne];
        let mut k = vec![0.0f32; seq * kv_dim];
        let mut v = vec![0.0f32; seq * kv_dim];
        let mut attn_scores = vec![0.0f32; seq];
        let mut attn_out = vec![0.0f32; seq * ne];
        let mut tmp1 = vec![0.0f32; seq * c.n_ff.max(ne)];
        let mut tmp2 = vec![0.0f32; seq * c.n_ff.max(ne)];

        for (li, layer) in self.layers.iter().enumerate() {
            rmsnorm_into(&x, &mut normed, seq, ne, &layer.attn_norm, c.rms_eps);

            matmul_rows(&layer.wq, ne, ne, &normed, seq, &mut q);
            matmul_rows(&layer.wk, kv_dim, ne, &normed, seq, &mut k);
            matmul_rows(&layer.wv, kv_dim, ne, &normed, seq, &mut v);
            add_bias_rows(&mut q, &layer.bq, seq, ne);
            add_bias_rows(&mut k, &layer.bk, seq, kv_dim);
            add_bias_rows(&mut v, &layer.bv, seq, kv_dim);
            if li == 0 {
                eprintln!("L0 attn_norm[0..4]={:?} len={}", &layer.attn_norm[0..4], layer.attn_norm.len());
                stats("L0 normed", &normed);
                eprintln!("L0 normed[0..6]={:?}", &normed[0..6]);
                eprintln!("L0 k[0..6]={:?} v[0..6]={:?}", &k[0..6], &v[0..6]);
            }
            if li == 0 || li == self.layers.len() - 1 {
                stats(&format!("L{li} q"), &q);
                stats(&format!("L{li} k"), &k);
                stats(&format!("L{li} v"), &v);
            }

            rope_norm(&mut q, seq, c.n_heads, hd, c.rope_theta);
            rope_norm(&mut k, seq, c.n_kv_heads, hd, c.rope_theta);

            let heads_per_kv = c.n_heads / c.n_kv_heads;
            let scale = 1.0 / (hd as f32).sqrt();
            for s in 0..seq {
                for h in 0..c.n_heads {
                    let kvh = h / heads_per_kv;
                    for t in 0..=s {
                        let q_off = (s * ne) + h * hd;
                        let k_off = (t * kv_dim) + kvh * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        attn_scores[t] = dot * scale;
                    }
                    softmax_row(&mut attn_scores[..=s]);
                    let o_off = (s * ne) + h * hd;
                    for d in 0..hd {
                        attn_out[o_off + d] = 0.0;
                    }
                    for t in 0..=s {
                        let w = attn_scores[t];
                        let v_off = (t * kv_dim) + kvh * hd;
                        for d in 0..hd {
                            attn_out[o_off + d] += w * v[v_off + d];
                        }
                    }
                }
            }

            matmul_rows(&layer.wo, ne, ne, &attn_out, seq, &mut tmp1);
            for i in 0..seq * ne {
                x[i] += tmp1[i];
            }
            if li == 0 || li == self.layers.len() - 1 {
                stats(&format!("L{li} dopo attn+res"), &x);
            }

            rmsnorm_into(&x, &mut normed, seq, ne, &layer.ffn_norm, c.rms_eps);
            let gate = &mut tmp1[..seq * c.n_ff];
            matmul_rows(&layer.w_gate, c.n_ff, ne, &normed, seq, gate);
            let up = &mut tmp2[..seq * c.n_ff];
            matmul_rows(&layer.w_up, c.n_ff, ne, &normed, seq, up);
            for i in 0..seq * c.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            let down = &mut tmp2[..seq * ne];
            matmul_rows(&layer.w_down, ne, c.n_ff, gate, seq, down);
            for i in 0..seq * ne {
                x[i] += down[i];
            }
            if li < 3 || li >= self.layers.len() - 2 {
                stats(&format!("L{li} dopo ffn+res"), &x);
            }
        }

        let mut last = vec![0.0f32; ne];
        last.copy_from_slice(&x[(seq - 1) * ne..seq * ne]);
        let last = rmsnorm_row(&last, &self.output_norm, c.rms_eps);
        stats("final norm", &last);

        let mut logits = vec![0.0f32; c.vocab_size];
        for o in 0..c.vocab_size {
            let w_off = o * ne;
            let mut sum = 0.0f32;
            for d in 0..ne {
                sum += self.output[w_off + d] * last[d];
            }
            logits[o] = sum;
        }
        stats("logits", &logits);
        logits
    }

    /// Forward: token IDs → logits [vocab].
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.config;
        let seq = tokens.len();
        let (ne, hd) = (c.n_embd, c.head_dim);
        let kv_dim = c.n_kv_heads * hd;

        // embedding lookup
        let mut x = vec![0.0f32; seq * ne];
        for (s, &t) in tokens.iter().enumerate() {
            let off = t as usize * ne;
            x[s * ne..(s + 1) * ne].copy_from_slice(&self.tok_embd[off..off + ne]);
        }

        let mut normed = vec![0.0f32; seq * ne];
        let mut q = vec![0.0f32; seq * ne];
        let mut k = vec![0.0f32; seq * kv_dim];
        let mut v = vec![0.0f32; seq * kv_dim];
        let mut attn_scores = vec![0.0f32; seq];
        let mut attn_out = vec![0.0f32; seq * ne];
        let mut tmp1 = vec![0.0f32; seq * c.n_ff.max(ne)];
        let mut tmp2 = vec![0.0f32; seq * c.n_ff.max(ne)];

        for (li, layer) in self.layers.iter().enumerate() {
            // ── attention ──
            // normed = rmsnorm(x) * attn_norm
            rmsnorm_into(&x, &mut normed, seq, ne, &layer.attn_norm, c.rms_eps);

            // q,k,v = normed @ W^T + b
            matmul_rows(&layer.wq, ne, ne, &normed, seq, &mut q);
            matmul_rows(&layer.wk, kv_dim, ne, &normed, seq, &mut k);
            matmul_rows(&layer.wv, kv_dim, ne, &normed, seq, &mut v);
            add_bias_rows(&mut q, &layer.bq, seq, ne);
            add_bias_rows(&mut k, &layer.bk, seq, kv_dim);
            add_bias_rows(&mut v, &layer.bv, seq, kv_dim);

            // RoPE (NORM: coppie interleaved) su q e k
            rope_norm(&mut q, seq, c.n_heads, hd, c.rope_theta);
            rope_norm(&mut k, seq, c.n_kv_heads, hd, c.rope_theta);

            // attenzione causale per testa (GQA: kv condivisi tra gruppi di teste)
            let heads_per_kv = c.n_heads / c.n_kv_heads;
            let scale = 1.0 / (hd as f32).sqrt();
            for s in 0..seq {
                for h in 0..c.n_heads {
                    let kvh = h / heads_per_kv;
                    // punteggi s vs t<=s
                    for t in 0..=s {
                        let q_off = (s * ne) + h * hd;
                        let k_off = (t * kv_dim) + kvh * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        attn_scores[t] = dot * scale;
                    }
                    softmax_row(&mut attn_scores[..=s]);
                    // somma pesata dei v
                    let o_off = (s * ne) + h * hd;
                    for d in 0..hd {
                        attn_out[o_off + d] = 0.0;
                    }
                    for t in 0..=s {
                        let w = attn_scores[t];
                        let v_off = (t * kv_dim) + kvh * hd;
                        for d in 0..hd {
                            attn_out[o_off + d] += w * v[v_off + d];
                        }
                    }
                }
            }

            // proiezione output + residuo
            matmul_rows(&layer.wo, ne, ne, &attn_out, seq, &mut tmp1);
            for i in 0..seq * ne {
                x[i] += tmp1[i];
            }

            // ── FFN ──
            rmsnorm_into(&x, &mut normed, seq, ne, &layer.ffn_norm, c.rms_eps);
            // gate e up
            let gate = &mut tmp1[..seq * c.n_ff];
            matmul_rows(&layer.w_gate, c.n_ff, ne, &normed, seq, gate);
            let up = &mut tmp2[..seq * c.n_ff];
            matmul_rows(&layer.w_up, c.n_ff, ne, &normed, seq, up);
            // silu(gate) * up
            for i in 0..seq * c.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            // down + residuo
            let down = &mut tmp2[..seq * ne];
            matmul_rows(&layer.w_down, ne, c.n_ff, gate, seq, down);
            for i in 0..seq * ne {
                x[i] += down[i];
            }

            let _ = li;
        }

        // ── finale ──
        let mut last = vec![0.0f32; ne];
        last.copy_from_slice(&x[(seq - 1) * ne..seq * ne]);
        let last = rmsnorm_row(&last, &self.output_norm, c.rms_eps);

        // logits = last @ output^T
        let mut logits = vec![0.0f32; c.vocab_size];
        for o in 0..c.vocab_size {
            let w_off = o * ne;
            let mut sum = 0.0f32;
            for d in 0..ne {
                sum += self.output[w_off + d] * last[d];
            }
            logits[o] = sum;
        }
        logits
    }
}

// ── primitive ──

/// RMSNorm: ogni riga (seq × n) di `x` normalizzata × pesi, scritta in `out`.
fn rmsnorm_into(x: &[f32], out: &mut [f32], seq: usize, n: usize, w: &[f32], eps: f32) {
    for s in 0..seq {
        let row = &x[s * n..(s + 1) * n];
        let mut ss = 0.0f32;
        for &v in row {
            ss += v * v;
        }
        ss /= n as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for i in 0..n {
            out[s * n + i] = row[i] * inv * w[i];
        }
    }
}

/// y(seq×n_out) = x(seq×n_in) @ W(n_out×n_in)^T — naive.
fn matmul_rows(w: &[f32], n_out: usize, n_in: usize, x: &[f32], seq: usize, y: &mut [f32]) {
    for s in 0..seq {
        for o in 0..n_out {
            let w_off = o * n_in;
            let mut sum = 0.0f32;
            for d in 0..n_in {
                sum += w[w_off + d] * x[s * n_in + d];
            }
            y[s * n_out + o] = sum;
        }
    }
}

fn add_bias_rows(y: &mut [f32], b: &[f32], seq: usize, n: usize) {
    for s in 0..seq {
        for i in 0..n {
            y[s * n + i] += b[i];
        }
    }
}

/// RoPE stile NEOX (half-split): coppia (i, i+hd/2) ruotata di angle = pos * theta^(-2i/d).
/// Verificato empiricamente su Qwen2.5: top-5 logits identico a llama.cpp
/// (lo stile interleaved/NORM dà top-1 diverso già dal primo token generato).
fn rope_norm(x: &mut [f32], seq: usize, n_heads: usize, hd: usize, theta: f32) {
    let half = hd / 2;
    for s in 0..seq {
        for h in 0..n_heads {
            let base = (s * n_heads + h) * hd;
            for i in 0..half {
                let freq = (theta).powf(-(2.0 * i as f32) / hd as f32);
                let angle = s as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * cos - x1 * sin;
                x[base + i + half] = x0 * sin + x1 * cos;
            }
        }
    }
}

fn softmax_row(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMSNorm su un singolo vettore (per il final norm).
fn rmsnorm_row(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}
