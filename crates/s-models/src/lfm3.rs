//! Architettura lfm3 (Liquid AI, tri-hybrid) — F10 finestra 1.
//!
//! Struttura MISURATA sull'GGUF ufficiale `LFM3-1.7B-F16.gguf` (header letto
//! via range request, 27/10/2026 — non dedotta dalle model card):
//!
//! - `lfm3.block_count = 24`
//! - `lfm3.attention.head_count_kv` = array per-layer, `0` sui layer NON attention
//!   → `[0,0,4,0,4,...]`: 13 layer a valore 0 (shortconv) + 11 attention
//! - `lfm3.attention.sliding_window_pattern` = array per-layer di bool
//!   → True su 5 layer (4, 8, 12, 16, 20) = sliding; gli altri 6 attention = full
//! - `lfm3.attention.sliding_window = 256` (finestra vera: 256 token)
//! - `lfm3.shortconv.l_cache = 7` (stato conv per layer: 7 righe)
//! - **due RoPE distinte**: `lfm3.rope.freq_base = 1e6` sui full,
//!   `lfm3.rope.freq_base_swa = 1e4` sugli sliding. Se non si separano,
//!   la parity salta in silenzio (errore piccolo, non un crash).
//!
//! Layout tensori (verificato sull'header):
//! - layer shortconv: `shortconv.conv[7,2048]` + `shortconv.in_proj[2048,6144]`
//!   + `shortconv.out_proj[2048,2048]` (+ 4 norm, ffn)
//! - layer attention: `attn_q[2048,2048]`, `attn_k[2048,512]`, `attn_v[2048,512]`,
//!   `attn_output[2048,2048]`, **`attn_gate[2048,16]`** (gate per testa),
//!   `attn_q_norm[128]`, `attn_k_norm[128]` (+ 4 norm, ffn)
//! - globali: `token_embd[2048,128000]`, `output_norm[2048]`, `rope_freqs[64]`
//!
//! STATO: config + classificazione dei layer + i due blocchi di forward
//! isolati (F10-6 shortconv, F10-7 attenzione) + test. Manca il `Model`
//! completo (embedding, MLP, residui, output head), quindi `Famiglia::Lfm3`
//! resta fuori da `registry::Famiglia::from_arch` finché la parity non
//! esiste — regola del registro: "mai per speranza". Il server rifiuta
//! `lfm3` con errore esplicito.

use s_gguf::{GgufFile, GgufValue, TensorInfo};

/// Tipo di layer nella struttura tri-hybrid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated short convolution: nessuna attenzione (`head_count_kv == 0`).
    ShortConv,
    /// Attenzione con finestra limitata (`sliding_window_pattern == true`).
    SlidingAttn,
    /// Attenzione globale (KV completo).
    FullAttn,
}

/// Config LFM3 letta dal metadata GGUF.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    /// Numero KV heads degli strati attention (il massimo dell'array per-layer).
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// RoPE dei layer full attention (1e6 su LFM3-1.7B).
    pub rope_theta: f32,
    /// RoPE dei layer sliding attention (1e4 su LFM3-1.7B). **Diversa dalla prima.**
    pub rope_theta_swa: f32,
    pub rope_n_rot: usize,
    pub rms_eps: f32,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub conv_l_cache: usize,
    /// `head_count_kv` per layer: 0 = layer shortconv (non attention).
    pub kv_heads_per_layer: Vec<usize>,
    /// `sliding_window_pattern` per layer (significativo solo dove kv > 0).
    pub sliding_pattern: Vec<bool>,
}

/// Legge un valore dal metadata per chiave esatta.
fn meta<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a GgufValue> {
    gguf.metadata
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
}

fn as_usize(v: &GgufValue) -> Option<usize> {
    match v {
        GgufValue::U8(x) => Some(*x as usize),
        GgufValue::U16(x) => Some(*x as usize),
        GgufValue::U32(x) => Some(*x as usize),
        GgufValue::U64(x) => Some(*x as usize),
        GgufValue::I8(x) => Some(*x as usize),
        GgufValue::I16(x) => Some(*x as usize),
        GgufValue::I32(x) => Some(*x as usize),
        GgufValue::I64(x) => Some(*x as usize),
        _ => None,
    }
}

/// Estrae un array di interi (tipo GGUF 9 con elementi u32/u64).
/// Estratto come funzione pura per poterlo testare senza un file GGUF.
pub fn parse_u_array(v: &GgufValue, what: &str) -> Result<Vec<usize>, String> {
    let GgufValue::Array(items) = v else {
        return Err(format!("{what}: atteso array, trovato {v:?}"));
    };
    items
        .iter()
        .map(|it| as_usize(it).ok_or_else(|| format!("{what}: elemento non intero ({it:?})")))
        .collect()
}

/// Estrae un array di bool (tipo GGUF 9 con elementi bool).
pub fn parse_bool_array(v: &GgufValue, what: &str) -> Result<Vec<bool>, String> {
    let GgufValue::Array(items) = v else {
        return Err(format!("{what}: atteso array, trovato {v:?}"));
    };
    items
        .iter()
        .map(|it| match it {
            GgufValue::Bool(b) => Ok(*b),
            // Alcuni convertitori emettono 0/1 come interi.
            other => as_usize(other)
                .map(|n| n != 0)
                .ok_or_else(|| format!("{what}: elemento non booleano ({other:?})")),
        })
        .collect()
}

impl Config {
    /// Config dal metadata GGUF (chiavi `lfm3.*`).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let get_u = |suffix: &str| -> Result<usize, String> {
            let key = format!("lfm3.{suffix}");
            meta(gguf, &key)
                .and_then(as_usize)
                .ok_or(format!("metadata mancante o non intero: {key}"))
        };
        let get_f = |suffix: &str| -> Result<f32, String> {
            let key = format!("lfm3.{suffix}");
            match meta(gguf, &key) {
                Some(GgufValue::F32(x)) => Ok(*x),
                Some(GgufValue::F64(x)) => Ok(*x as f32),
                Some(other) => Err(format!("metadata {key}: atteso float, trovato {other:?}")),
                None => Err(format!("metadata mancante: {key}")),
            }
        };

        let n_layers = get_u("block_count")?;
        let n_embd = get_u("embedding_length")?;
        let n_heads = get_u("attention.head_count")?;
        let head_dim = get_u("attention.key_length")?;

        let kv_heads_per_layer = parse_u_array(
            meta(gguf, "lfm3.attention.head_count_kv")
                .ok_or("metadata mancante: lfm3.attention.head_count_kv")?,
            "lfm3.attention.head_count_kv",
        )?;
        let sliding_pattern = parse_bool_array(
            meta(gguf, "lfm3.attention.sliding_window_pattern")
                .ok_or("metadata mancante: lfm3.attention.sliding_window_pattern")?,
            "lfm3.attention.sliding_window_pattern",
        )?;

        if kv_heads_per_layer.len() != n_layers {
            return Err(format!(
                "lfm3: head_count_kv ha {} elementi ma block_count è {n_layers}",
                kv_heads_per_layer.len()
            ));
        }
        if sliding_pattern.len() != n_layers {
            return Err(format!(
                "lfm3: sliding_window_pattern ha {} elementi ma block_count è {n_layers}",
                sliding_pattern.len()
            ));
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);
        if n_kv_heads == 0 {
            return Err("lfm3: nessun layer con head_count_kv > 0".into());
        }

        let rope_n_rot = {
            let key = "lfm3.rope.dimension_count";
            meta(gguf, key).and_then(as_usize).unwrap_or(head_dim)
        };

        Ok(Self {
            n_layers,
            n_embd,
            n_ff: get_u("feed_forward_length")?,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta: get_f("rope.freq_base")?,
            rope_theta_swa: get_f("rope.freq_base_swa")?,
            rope_n_rot,
            rms_eps: get_f("attention.layer_norm_rms_epsilon")?,
            vocab_size: gguf
                .tensor("token_embd.weight")
                .map_err(|e| e.to_string())?
                .dims[1] as usize,
            sliding_window: get_u("attention.sliding_window")?,
            conv_l_cache: get_u("shortconv.l_cache")?,
            kv_heads_per_layer,
            sliding_pattern,
        })
    }

    /// Classifica il layer `li` (0-based).
    pub fn layer_kind(&self, li: usize) -> Result<LayerKind, String> {
        let kv = *self
            .kv_heads_per_layer
            .get(li)
            .ok_or_else(|| format!("lfm3: layer {li} fuori range ({} layer)", self.n_layers))?;
        if kv == 0 {
            Ok(LayerKind::ShortConv)
        } else if *self
            .sliding_pattern
            .get(li)
            .ok_or_else(|| format!("lfm3: sliding_pattern corto al layer {li}"))?
        {
            Ok(LayerKind::SlidingAttn)
        } else {
            Ok(LayerKind::FullAttn)
        }
    }

    /// Tutti i layer classificati, in ordine.
    pub fn layer_kinds(&self) -> Result<Vec<LayerKind>, String> {
        (0..self.n_layers).map(|i| self.layer_kind(i)).collect()
    }

    /// Quanti layer per tipo: (shortconv, sliding, full).
    pub fn kind_counts(&self) -> Result<(usize, usize, usize), String> {
        let mut c = (0, 0, 0);
        for k in self.layer_kinds()? {
            match k {
                LayerKind::ShortConv => c.0 += 1,
                LayerKind::SlidingAttn => c.1 += 1,
                LayerKind::FullAttn => c.2 += 1,
            }
        }
        Ok(c)
    }

    /// La RoPE da usare per un dato layer attention.
    /// `None` sui layer shortconv (nessuna RoPE).
    pub fn rope_theta_for(&self, li: usize) -> Result<Option<f32>, String> {
        Ok(match self.layer_kind(li)? {
            LayerKind::ShortConv => None,
            LayerKind::SlidingAttn => Some(self.rope_theta_swa),
            LayerKind::FullAttn => Some(self.rope_theta),
        })
    }

    /// Numero di frequenze RoPE ruotate per il layer `li`, LETTO dal GGUF:
    /// `lfm3.rope.dimension_count / 2` (64 sul modello reale, dove il
    /// dimension_count è 128).
    ///
    /// ATTENZIONE: è il massimo a rotazione piena. Il `partial_rotary_factor`
    /// di HF (0.5 sui layer full, verificato in `config.json`) NON è nel
    /// metadata GGUF: è codificato in `rope_freqs.weight` (misurato: 32
    /// valori 1.0 + 32 valori 1e30 sui full). Chi costruisce un
    /// [`AttentionBlock`] full deve quindi passare 32 e non il valore di
    /// questo metodo (vedi la nota F10-7).
    pub fn rope_angles_for(&self, li: usize) -> Result<usize, String> {
        Ok(match self.layer_kind(li)? {
            LayerKind::ShortConv => 0,
            LayerKind::SlidingAttn | LayerKind::FullAttn => self.rope_n_rot / 2,
        })
    }
}

/// Pesi di un layer shortconv (gated convolution, nessuna attenzione).
#[derive(Debug, Clone)]
pub struct ShortConvWeights {
    pub conv: TensorInfo,    // [l_cache, n_embd] = [7, 2048]
    pub in_proj: TensorInfo, // [n_embd, 3*n_embd] = [2048, 6144]
    pub out_proj: TensorInfo, // [n_embd, n_embd] = [2048, 2048]
}

/// Pesi di un layer attention (GQA + gate per testa + Q/K norm).
#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub q: TensorInfo,      // [n_embd, n_embd] = [2048, 2048]
    pub k: TensorInfo,      // [n_embd, n_kv_heads*head_dim] = [2048, 512]
    pub v: TensorInfo,      // [n_embd, n_kv_heads*head_dim] = [2048, 512]
    pub output: TensorInfo, // [n_embd, n_embd] = [2048, 2048]
    pub gate: TensorInfo,   // [n_embd, n_heads] = [2048, 16] (gate per testa)
    pub q_norm: TensorInfo, // [head_dim] = [128]
    pub k_norm: TensorInfo, // [head_dim] = [128]
}

/// Variante di layer: solo uno dei due set esiste per ogni indice.
#[derive(Debug, Clone)]
pub enum LayerVariant {
    ShortConv(ShortConvWeights),
    Attention(AttnWeights),
}

/// Pesi di un layer: 4 norm + FFN (comuni) + variante shortconv/attention.
#[derive(Debug, Clone)]
pub struct Lfm3LayerWeights {
    pub kind: LayerKind,
    pub attn_norm: TensorInfo,            // [n_embd]
    pub post_attention_norm: TensorInfo, // [n_embd]
    pub ffn_norm: TensorInfo,             // [n_embd]
    pub post_ffw_norm: TensorInfo,       // [n_embd]
    pub ffn_gate: TensorInfo,            // [n_embd, n_ff]
    pub ffn_up: TensorInfo,              // [n_embd, n_ff]
    pub ffn_down: TensorInfo,            // [n_ff, n_embd]
    pub variant: LayerVariant,
}

/// Pesi LFM3 verificati contro l'header GGUF.
///
/// Solo metadati dei tensori (nomi + forme + tipi): nessuna dequantizzazione,
/// nessun forward. Il forward verrà dopo (finestra successiva), su forme già
/// garantite qui.
#[derive(Debug, Clone)]
pub struct Lfm3Weights {
    pub layers: Vec<Lfm3LayerWeights>,
    pub token_embd: TensorInfo, // [n_embd, vocab] = [2048, 128000]
    pub output_norm: TensorInfo, // [n_embd] = [2048]
    pub rope_freqs: TensorInfo, // [rope_n_rot/2] = [64]
}

/// Verifica che il tensore `name` esista e abbia esattamente `expected`.
/// Ogni fallimento nomina tensore, forma attesa e forma trovata.
fn expect_dims(gguf: &GgufFile, name: &str, expected: &[u64]) -> Result<TensorInfo, String> {
    let info = gguf
        .tensor(name)
        .map_err(|e| format!("{name}: {e} (atteso {expected:?})"))?;
    if info.dims != expected {
        return Err(format!(
            "{name}: forma attesa {expected:?}, trovata {:?} (tipo {:?})",
            info.dims, info.ggml_type
        ));
    }
    Ok(info.clone())
}

impl Lfm3Weights {
    /// Carica (e verifica) tutti i pesi dal file GGUF secondo `cfg`.
    ///
    /// Per ogni layer sceglie il set giusto da `cfg.layer_kind(li)`:
    /// shortconv → `shortconv.*`, attention → `attn_*`. Nessun `unwrap`:
    /// ogni errore nomina tensore, forma attesa e forma trovata.
    pub fn load(gguf: &GgufFile, cfg: &Config) -> Result<Self, String> {
        let ne = cfg.n_embd as u64;
        let nff = cfg.n_ff as u64;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let hd = cfg.head_dim as u64;
        let nh = cfg.n_heads as u64;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let kind = cfg.layer_kind(li)?;
            let base = format!("blk.{li}.");
            let t = |suffix: &str, expected: &[u64]| {
                expect_dims(gguf, &format!("{base}{suffix}"), expected)
            };

            let attn_norm = t("attn_norm.weight", &[ne])?;
            let post_attention_norm = t("post_attention_norm.weight", &[ne])?;
            let ffn_norm = t("ffn_norm.weight", &[ne])?;
            let post_ffw_norm = t("post_ffw_norm.weight", &[ne])?;
            let ffn_gate = t("ffn_gate.weight", &[ne, nff])?;
            let ffn_up = t("ffn_up.weight", &[ne, nff])?;
            let ffn_down = t("ffn_down.weight", &[nff, ne])?;

            let variant = match kind {
                LayerKind::ShortConv => LayerVariant::ShortConv(ShortConvWeights {
                    conv: t("shortconv.conv.weight", &[cfg.conv_l_cache as u64, ne])?,
                    in_proj: t("shortconv.in_proj.weight", &[ne, 3 * ne])?,
                    out_proj: t("shortconv.out_proj.weight", &[ne, ne])?,
                }),
                LayerKind::SlidingAttn | LayerKind::FullAttn => {
                    LayerVariant::Attention(AttnWeights {
                        q: t("attn_q.weight", &[ne, ne])?,
                        k: t("attn_k.weight", &[ne, kv_dim])?,
                        v: t("attn_v.weight", &[ne, kv_dim])?,
                        output: t("attn_output.weight", &[ne, ne])?,
                        gate: t("attn_gate.weight", &[ne, nh])?,
                        q_norm: t("attn_q_norm.weight", &[hd])?,
                        k_norm: t("attn_k_norm.weight", &[hd])?,
                    })
                }
            };

            layers.push(Lfm3LayerWeights {
                kind,
                attn_norm,
                post_attention_norm,
                ffn_norm,
                post_ffw_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                variant,
            });
        }

        let token_embd = expect_dims(gguf, "token_embd.weight", &[ne, cfg.vocab_size as u64])?;
        let output_norm = expect_dims(gguf, "output_norm.weight", &[ne])?;
        let rope_freqs = expect_dims(gguf, "rope_freqs.weight", &[(cfg.rope_n_rot as u64) / 2])?;

        Ok(Self { layers, token_embd, output_norm, rope_freqs })
    }

    /// Quanti layer per tipo: (shortconv, attention, sliding).
    /// Lo sliding è un sottoinsieme di attention (finestra 256 vs globale).
    pub fn kind_counts(&self) -> (usize, usize, usize) {
        let mut short = 0;
        let mut sliding = 0;
        let mut full = 0;
        for l in &self.layers {
            match l.kind {
                LayerKind::ShortConv => short += 1,
                LayerKind::SlidingAttn => sliding += 1,
                LayerKind::FullAttn => full += 1,
            }
        }
        (short, sliding + full, sliding)
    }

    /// Indici dei layer sliding attention (finestra limitata).
    pub fn sliding_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.kind == LayerKind::SlidingAttn)
            .map(|(i, _)| i)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// F10-6 — forward del blocco shortconv (SOLO il blocco).
//
// Formula da `Lfm3ShortConv.forward` di `LiquidAI/LFM3-1.7B`
// (`modeling_lfm3.py`, verificata il 28/10/2026 sul sorgente ufficiale):
//
// ```python
// BCx = self.in_proj(hidden_states)      # [B, T, 3*hidden]
// B, C, x = BCx.chunk(3, dim=-1)
// h = B * x                              # gate B PRIMA della conv, nessuna attivazione
// h = causal_depthwise_conv(h, k=7)      # groups=hidden, padding=k-1, troncata a seq_len
// h = C * h                              # secondo gate DOPO la conv
// y = self.out_proj(h)
// ```
//
// Quindi `out = out_proj(C * conv_k7(B * x))` con `B, C, x = chunk(in_proj(h), 3)`.
// Config ufficiale (`config.json`): `conv_bias = False`, `conv_L_cache = 7`.
// Buffer di decode (`causal_conv1d_update`): i valori precedenti di `B * x`
// (l'ingresso della conv, NON l'uscita), lunghezza `l_cache - 1 = 6`.
//
// NON c'è qui: MLP, embedding, `Model` completo, dispatch nel
// server, `Famiglia::Lfm3`, CUDA. Solo il blocco.
// ═══════════════════════════════════════════════════════════════════

/// Pesi densi f32 del blocco shortconv, pronti per il forward CPU.
///
/// Tenuto separato da [`ShortConvWeights`] di proposito: quello è il
/// descrittore dei tensori GGUF usato da `Lfm3Weights::load` (finestra T003,
/// solo metadati + forme, nessuna dequantizzazione), mentre qui servono i
/// valori. Il futuro `Model` LFM3 dequantizzerà i tensori in questo formato.
///
/// Layout (stesso ordine dei byte del GGUF dequantizzato, le cui dims sono
/// `[in, out]` — vedi `matmul_row_vec` in `qwen35.rs` per la convenzione):
/// - `in_proj`: `[3*n_embd, n_embd]` in C-order,
///   `BCx[j] = Σ_i h[i]·in_proj[j*n_embd + i]`
/// - `conv`: `[n_embd, l_cache]` in C-order, `conv[c*l_cache + k]` dove `k = 0`
///   è il tap più vecchio (6 passi fa) e `k = l_cache - 1` il corrente —
///   come `torch.einsum("bdtk,dk->bdt", windows, weight)` della reference
/// - `out_proj`: `[n_embd, n_embd]` in C-order, stessa convenzione di `in_proj`
#[derive(Debug, Clone)]
pub struct ShortConvBlock {
    pub n_embd: usize,
    pub l_cache: usize,
    pub in_proj: Vec<f32>,
    pub conv: Vec<f32>,
    pub out_proj: Vec<f32>,
}

impl ShortConvBlock {
    /// Costruisce il blocco verificando le forme.
    ///
    /// Niente `unwrap` sui dati in input: ogni errore nomina il tensore,
    /// la lunghezza attesa e quella trovata.
    pub fn new(
        n_embd: usize,
        l_cache: usize,
        in_proj: Vec<f32>,
        conv: Vec<f32>,
        out_proj: Vec<f32>,
    ) -> Result<Self, String> {
        if n_embd == 0 {
            return Err("shortconv: n_embd deve essere > 0".into());
        }
        if l_cache < 2 {
            return Err(format!(
                "shortconv: l_cache deve essere >= 2 (storia + corrente), trovato {l_cache}"
            ));
        }
        if in_proj.len() != 3 * n_embd * n_embd {
            return Err(format!(
                "shortconv in_proj: attesi {} elementi ([3*{n_embd}, {n_embd}]), trovati {}",
                3 * n_embd * n_embd,
                in_proj.len()
            ));
        }
        if conv.len() != n_embd * l_cache {
            return Err(format!(
                "shortconv conv: attesi {} elementi ([{n_embd}, {l_cache}]), trovati {}",
                n_embd * l_cache,
                conv.len()
            ));
        }
        if out_proj.len() != n_embd * n_embd {
            return Err(format!(
                "shortconv out_proj: attesi {} elementi ([{n_embd}, {n_embd}]), trovati {}",
                n_embd * n_embd,
                out_proj.len()
            ));
        }
        Ok(Self { n_embd, l_cache, in_proj, conv, out_proj })
    }

    /// Righe di storia tenute in `conv_buf`: `l_cache - 1` (6 sul modello reale).
    pub fn hist_rows(&self) -> usize {
        self.l_cache - 1
    }

    /// Una posizione: `h [n_embd]` → `out [n_embd]`, aggiornando il buffer conv.
    ///
    /// `conv_buf` tiene le ultime `l_cache - 1` righe di `B * x` (l'ingresso
    /// della conv, come `conv_state` in `causal_conv1d_update`): la conv legge
    /// `[buf(6 righe)…, corrente]` col tap `k = 0` sul più vecchio, poi il
    /// buffer avanza di una riga (scarta la più vecchia, accoda `B * x`).
    pub fn forward_step(&self, h: &[f32], conv_buf: &mut [f32], out: &mut [f32]) {
        let ne = self.n_embd;
        let kk = self.l_cache;
        let hr = self.hist_rows();
        assert_eq!(h.len(), ne, "shortconv step: h ha {} elementi, attesi {ne}", h.len());
        assert_eq!(
            conv_buf.len(),
            hr * ne,
            "shortconv step: conv_buf ha {} elementi, attesi {}",
            conv_buf.len(),
            hr * ne
        );
        assert_eq!(
            out.len(),
            ne,
            "shortconv step: out ha {} elementi, attesi {ne}",
            out.len()
        );

        // 1. BCx = in_proj(h), poi chunk in B, C, x.
        let mut bcx = vec![0.0f32; 3 * ne];
        matvec(&self.in_proj, ne, 3 * ne, h, &mut bcx);
        let (b, rest) = bcx.split_at(ne);
        let (c, x) = rest.split_at(ne);

        // 2. Gate B PRIMA della conv, nessuna attivazione (reference: `h = B * x`).
        let mut g = vec![0.0f32; ne];
        for i in 0..ne {
            g[i] = b[i] * x[i];
        }

        // 3. Conv causale depthwise su [buf…, g]: tap k=0 sul più vecchio.
        let mut cv = vec![0.0f32; ne];
        for ci in 0..ne {
            let mut sum = 0.0f32;
            for k2 in 0..hr {
                sum += self.conv[ci * kk + k2] * conv_buf[k2 * ne + ci];
            }
            sum += self.conv[ci * kk + hr] * g[ci];
            cv[ci] = sum;
        }

        // 4. Gate C DOPO la conv + out_proj (reference: `y = C * hidden_states`).
        let mut y2 = vec![0.0f32; ne];
        for i in 0..ne {
            y2[i] = c[i] * cv[i];
        }
        matvec(&self.out_proj, ne, ne, &y2, out);

        // 5. Avanza la storia: scarta la riga più vecchia, accoda B * x.
        if hr > 0 {
            conv_buf.copy_within(ne.., 0);
            conv_buf[(hr - 1) * ne..].copy_from_slice(&g);
        }
    }

    /// Prefill: T posizioni in un colpo.
    ///
    /// `h`/`out` sono flat `[T, n_embd]` in row-major, `conv_buf` come in
    /// [`ShortConvBlock::forward_step`]. Percorso separato dallo step
    /// (storia locale + loop a sequenza invece di storia persistente):
    /// l'accordo fra i due è la proprietà che serve al motore (test 2).
    /// Alla fine `conv_buf` contiene le ultime `l_cache - 1` righe di `B * x`,
    /// pronto per proseguire in decode.
    pub fn forward_seq(&self, h: &[f32], t: usize, conv_buf: &mut [f32], out: &mut [f32]) {
        let ne = self.n_embd;
        let kk = self.l_cache;
        let hr = self.hist_rows();
        assert_eq!(
            h.len(),
            t * ne,
            "shortconv seq: h ha {} elementi, attesi {t}*{ne}",
            h.len()
        );
        assert_eq!(
            out.len(),
            t * ne,
            "shortconv seq: out ha {} elementi, attesi {t}*{ne}",
            out.len()
        );
        assert_eq!(
            conv_buf.len(),
            hr * ne,
            "shortconv seq: conv_buf ha {} elementi, attesi {}",
            conv_buf.len(),
            hr * ne
        );
        if t == 0 {
            return;
        }

        // Storia dei gate B*x: [buf_old (hr righe) | g_0 … g_{t-1}].
        let mut hist = vec![0.0f32; (hr + t) * ne];
        hist[..hr * ne].copy_from_slice(conv_buf);

        for p in 0..t {
            let hp = &h[p * ne..(p + 1) * ne];

            // 1-2. in_proj + chunk + gate B (loop propri di questo percorso).
            let mut bcx = vec![0.0f32; 3 * ne];
            for j in 0..3 * ne {
                let mut sum = 0.0f32;
                for i in 0..ne {
                    sum += hp[i] * self.in_proj[j * ne + i];
                }
                bcx[j] = sum;
            }
            for i in 0..ne {
                hist[(hr + p) * ne + i] = bcx[i] * bcx[2 * ne + i];
            }

            // 3. Conv causale: finestra di `l_cache` righe che termina in p.
            let mut cv = vec![0.0f32; ne];
            for ci in 0..ne {
                let mut sum = 0.0f32;
                for k2 in 0..kk {
                    sum += self.conv[ci * kk + k2] * hist[(p + k2) * ne + ci];
                }
                cv[ci] = sum;
            }

            // 4. Gate C + out_proj.
            let op = &mut out[p * ne..(p + 1) * ne];
            for j in 0..ne {
                let mut sum = 0.0f32;
                for i in 0..ne {
                    sum += bcx[ne + i] * cv[i] * self.out_proj[j * ne + i];
                }
                op[j] = sum;
            }
        }

        // Ultime hr righe di storia → conv_buf (decode può proseguire da qui).
        conv_buf.copy_from_slice(&hist[t * ne..(t + hr) * ne]);
    }
}

/// `m` in C-order `[rows, cols]`: `y[j] = Σ_i x[i]·m[j*cols + i]`.
/// Stessa convenzione di `matmul_row_vec` in `qwen35.rs` (ordine dei byte GGUF).
fn matvec(m: &[f32], cols: usize, rows: usize, x: &[f32], y: &mut [f32]) {
    for j in 0..rows {
        let mut sum = 0.0f32;
        for i in 0..cols {
            sum += x[i] * m[j * cols + i];
        }
        y[j] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════════
// F10-7 — forward dell'ATTENZIONE (solo il blocco).
//
// Formula da `Lfm3Attention.forward` di `LiquidAI/LFM3-1.7B`
// (`modeling_lfm3.py`, verificata il 03/11/2026 sul sorgente ufficiale):
//
// ```python
// q = q_norm(q_proj(h).view(..., n_heads, head_dim))      # RMSNorm per testa
// k = k_norm(k_proj(h).view(..., n_kv_heads, head_dim))
// v = v_proj(h).view(..., n_kv_heads, head_dim)
// q, k = apply_rotary_pos_emb(q, k, cos, sin)             # f32, rotate_half
// o = attenzione_causale(q, k, v)                         # GQA, scala head_dim^-0.5
// o = o * sigmoid(gate_proj(h))                           # gate PER TESTA
// out = o_proj(o)
// ```
//
// Particolari MISURATI, non dedotti:
// - RoPE `rotate_half` (convenzione HF): `[x1, x2] → [x1·cos − x2·sin,
//   x2·cos + x1·sin]`, non la variante interleaved.
// - `partial_rotary_factor = 0.5` sui layer full (`config.json` ufficiale):
//   su 64 frequenze ruotano solo le prime 32; le altre hanno `inv_freq = 0`
//   (angolo 0 ⇒ cos=1, sin=0 ⇒ dimensioni intatte). Il GGUF NON espone il
//   fattore nel metadata: lo codifica in `rope_freqs.weight` (32 valori 1.0
//   + 32 valori 1e30 sui full). Per questo il numero di angoli è un parametro
//   esplicito di [`AttentionSpec`]; `Config::rope_angles_for` dà il massimo.
// - Finestra sliding: la maschera HF è `kv_idx > q_idx − sliding_window`
//   (`transformers/masking_utils.py:99`), quindi il token a distanza esatta
//   `sliding_window` è ESCLUSO (256 chiavi, non 257). La formula del task
//   ("indice < pos − 256") ne terrebbe 257: qui si segue il riferimento.
//
// NON c'è qui: KV cache persistente su device, MLP, embedding, `Model`
// completo, dispatch nel server, `Famiglia::Lfm3`. Solo il blocco.
// ═══════════════════════════════════════════════════════════════════

/// Dimensioni e parametri del blocco attention (un layer).
///
/// `rope_angles` è il numero di frequenze ruotate: `head_dim/2` a rotazione
/// piena (layer sliding), 32 su 64 per i layer full (partial factor 0.5).
/// Le frequenze oltre `rope_angles` hanno inv_freq 0 e non ruotano.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSpec {
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_angles: usize,
    /// `None` = attenzione globale; `Some(w)` = sliding di `w` chiavi.
    pub finestra: Option<usize>,
}

/// Pesi densi f32 del blocco attention, pronti per il forward CPU.
///
/// Stessa convenzione di [`ShortConvBlock`] (ordine dei byte GGUF, dims
/// `[in, out]` → flat C-order `[out, in]`):
/// - `q`: `[n_heads*head_dim, n_embd]`
/// - `k`, `v`: `[n_kv_heads*head_dim, n_embd]`
/// - `output`: `[n_embd, n_heads*head_dim]`
/// - `gate`: `[n_heads, n_embd]`
/// - `q_norm`, `k_norm`: `[head_dim]`
#[derive(Debug, Clone)]
pub struct AttentionDense {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub output: Vec<f32>,
    pub gate: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Pesi densi + parametri del blocco attention.
#[derive(Debug, Clone)]
pub struct AttentionBlock {
    pub spec: AttentionSpec,
    pub pesi: AttentionDense,
}

impl AttentionBlock {
    /// Costruisce il blocco verificando tutte le forme.
    ///
    /// Niente `unwrap`: ogni errore nomina il tensore, la lunghezza attesa
    /// e quella trovata.
    pub fn new(spec: AttentionSpec, pesi: AttentionDense) -> Result<Self, String> {
        if spec.n_embd == 0 {
            return Err("attention: n_embd deve essere > 0".into());
        }
        if spec.head_dim == 0 || !spec.head_dim.is_multiple_of(2) {
            return Err(format!(
                "attention: head_dim deve essere pari e > 0, trovato {}",
                spec.head_dim
            ));
        }
        if spec.n_heads == 0 || spec.n_kv_heads == 0 {
            return Err("attention: n_heads e n_kv_heads devono essere > 0".into());
        }
        if !spec.n_heads.is_multiple_of(spec.n_kv_heads) {
            return Err(format!(
                "attention: n_heads ({}) non è multiplo di n_kv_heads ({})",
                spec.n_heads, spec.n_kv_heads
            ));
        }
        if spec.rope_angles > spec.head_dim / 2 {
            return Err(format!(
                "attention: rope_angles {} > head_dim/2 = {}",
                spec.rope_angles,
                spec.head_dim / 2
            ));
        }
        if matches!(spec.finestra, Some(0)) {
            return Err("attention: finestra sliding deve essere > 0".into());
        }

        let nq = spec.n_heads * spec.head_dim;
        let nkv = spec.n_kv_heads * spec.head_dim;
        let ne = spec.n_embd;
        let atteso = |nome: &str, len: usize, voluto: usize| -> Result<(), String> {
            if len != voluto {
                return Err(format!(
                    "attention {nome}: attesi {voluto} elementi, trovati {len}"
                ));
            }
            Ok(())
        };
        atteso("q", pesi.q.len(), ne * nq)?;
        atteso("k", pesi.k.len(), ne * nkv)?;
        atteso("v", pesi.v.len(), ne * nkv)?;
        atteso("output", pesi.output.len(), ne * nq)?;
        atteso("gate", pesi.gate.len(), ne * spec.n_heads)?;
        atteso("q_norm", pesi.q_norm.len(), spec.head_dim)?;
        atteso("k_norm", pesi.k_norm.len(), spec.head_dim)?;

        Ok(Self { spec, pesi })
    }

    /// Dimensione dell'uscita Q (teste × head_dim).
    pub fn n_q(&self) -> usize {
        self.spec.n_heads * self.spec.head_dim
    }

    /// Dimensione di K e V (KV head × head_dim).
    pub fn n_kv(&self) -> usize {
        self.spec.n_kv_heads * self.spec.head_dim
    }

    /// q, k, v per UNA posizione: proiezioni, RMSNorm per testa, RoPE.
    /// Esposta per i test sulle forme; il forward la usa internamente.
    pub fn calcola_qkv(&self, h: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let s = &self.spec;
        let ne = s.n_embd;
        let hd = s.head_dim;
        let nq = self.n_q();
        let nkv = self.n_kv();
        assert_eq!(h.len(), ne, "attention qkv: h ha {} elementi, attesi {ne}", h.len());

        // 1. Proiezioni.
        let mut q = vec![0.0f32; nq];
        matvec(&self.pesi.q, ne, nq, h, &mut q);
        let mut k = vec![0.0f32; nkv];
        matvec(&self.pesi.k, ne, nkv, h, &mut k);
        let mut v = vec![0.0f32; nkv];
        matvec(&self.pesi.v, ne, nkv, h, &mut v);

        // 2. RMSNorm per testa (q_norm / k_norm), prima della RoPE.
        for hh in 0..s.n_heads {
            rmsnorm_in_place(&mut q[hh * hd..(hh + 1) * hd], &self.pesi.q_norm, s.rms_eps);
        }
        for hh in 0..s.n_kv_heads {
            rmsnorm_in_place(&mut k[hh * hd..(hh + 1) * hd], &self.pesi.k_norm, s.rms_eps);
        }

        // 3. RoPE in f32 su q e k.
        rope_rotante(&mut q, s.n_heads, hd, s.rope_theta, s.rope_angles, pos);
        rope_rotante(&mut k, s.n_kv_heads, hd, s.rope_theta, s.rope_angles, pos);
        (q, k, v)
    }

    /// Attenzione causale (con eventuale finestra) per UNA query.
    ///
    /// `k`/`v` sono il prefisso `[t, n_kv]` con posizione assoluta della
    /// prima chiave `k_abs0`; `q_abs` è la posizione assoluta della query.
    /// Restituisce l'uscita per testa `[n_heads*head_dim]`, senza gate né
    /// `o_proj`. La visibilità segue il riferimento HF:
    /// `k_abs <= q_abs` (causale) e, con finestra `w`, `k_abs > q_abs − w`.
    fn attenzione_grezza(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        k_abs0: usize,
        q_abs: usize,
    ) -> Vec<f32> {
        let s = &self.spec;
        let hd = s.head_dim;
        let nq = self.n_q();
        let nkv = self.n_kv();
        assert_eq!(q.len(), nq, "attention: q ha {} elementi, attesi {nq}", q.len());
        assert_eq!(
            k.len(),
            t * nkv,
            "attention: k ha {} elementi, attesi {t}*{nkv}",
            k.len()
        );
        assert_eq!(
            v.len(),
            t * nkv,
            "attention: v ha {} elementi, attesi {t}*{nkv}",
            v.len()
        );

        let gruppo = s.n_heads / s.n_kv_heads;
        let scala = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; nq];
        let mut punteggi = vec![0.0f32; t];

        for hh in 0..s.n_heads {
            let kvh = hh / gruppo;
            let qb = &q[hh * hd..(hh + 1) * hd];

            // 1. score = (q·k) * head_dim^-0.5, con maschera causale/finestra.
            let mut max = f32::NEG_INFINITY;
            for j in 0..t {
                let k_abs = k_abs0 + j;
                let causale = k_abs <= q_abs;
                let in_finestra = match s.finestra {
                    None => true,
                    Some(w) => k_abs + w > q_abs,
                };
                if !(causale && in_finestra) {
                    punteggi[j] = f32::NEG_INFINITY;
                    continue;
                }
                let kb = &k[j * nkv + kvh * hd..j * nkv + kvh * hd + hd];
                let mut sum = 0.0f32;
                for d in 0..hd {
                    sum += qb[d] * kb[d];
                }
                let sc = sum * scala;
                punteggi[j] = sc;
                if sc > max {
                    max = sc;
                }
            }

            // 2. Softmax stabile (max-shift).
            let mut somma = 0.0f32;
            for p in punteggi.iter_mut() {
                if p.is_finite() {
                    *p = (*p - max).exp();
                    somma += *p;
                } else {
                    *p = 0.0;
                }
            }
            assert!(somma > 0.0, "attention: nessuna chiave visibile (query {q_abs})");

            // 3. Somma pesata su v.
            for j in 0..t {
                let w = punteggi[j] / somma;
                if w == 0.0 {
                    continue;
                }
                let vb = &v[j * nkv + kvh * hd..j * nkv + kvh * hd + hd];
                let ob = &mut out[hh * hd..(hh + 1) * hd];
                for d in 0..hd {
                    ob[d] += w * vb[d];
                }
            }
        }
        out
    }

    /// Gate sigmoideo per testa: `sigmoid(gate_proj(h))`, `[n_heads]`.
    fn calcola_gate(&self, h: &[f32]) -> Vec<f32> {
        let s = &self.spec;
        let mut g = vec![0.0f32; s.n_heads];
        matvec(&self.pesi.gate, s.n_embd, s.n_heads, h, &mut g);
        for x in g.iter_mut() {
            *x = sigmoid(*x);
        }
        g
    }

    /// Moltiplica l'uscita attention per il gate, testa per testa.
    /// Funzione separata perché il gate è moltiplicativo: con `gate` tutta a
    /// zero l'uscita si annulla esattamente.
    fn applica_gate(&self, attn: &mut [f32], gate: &[f32]) {
        let hd = self.spec.head_dim;
        assert_eq!(
            attn.len(),
            self.n_q(),
            "attention: attn ha {} elementi, attesi {}",
            attn.len(),
            self.n_q()
        );
        assert_eq!(
            gate.len(),
            self.spec.n_heads,
            "attention: gate ha {} elementi, attesi {}",
            gate.len(),
            self.spec.n_heads
        );
        for (hh, &g) in gate.iter().enumerate() {
            for x in &mut attn[hh * hd..(hh + 1) * hd] {
                *x *= g;
            }
        }
    }

    /// Gate + `o_proj` su una posizione.
    fn uscita(&self, h: &[f32], attn: &[f32], out: &mut [f32]) {
        let ne = self.spec.n_embd;
        assert_eq!(out.len(), ne, "attention: out ha {} elementi, attesi {ne}", out.len());
        let g = self.calcola_gate(h);
        let mut gated = attn.to_vec();
        self.applica_gate(&mut gated, &g);
        matvec(&self.pesi.output, self.n_q(), ne, &gated, out);
    }

    /// Decode: una posizione `h [n_embd]` → `out [n_embd]`.
    ///
    /// `k_cache`/`v_cache` sono flat `[pos, n_kv]` e devono contenere le
    /// posizioni `0..pos` (stesso contratto di una cache che parte da zero:
    /// `cache.len() == pos * n_kv`). Il token corrente viene accodato e
    /// l'attenzione vede tutto il prefisso (con finestra se sliding).
    pub fn forward_step(
        &self,
        h: &[f32],
        pos: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        let nkv = self.n_kv();
        assert_eq!(
            k_cache.len(),
            v_cache.len(),
            "attention step: k_cache ha {} elementi, v_cache {}",
            k_cache.len(),
            v_cache.len()
        );
        assert_eq!(
            k_cache.len(),
            pos * nkv,
            "attention step: la cache deve contenere pos={pos} chiavi ({} elementi), trovati {}",
            pos * nkv,
            k_cache.len()
        );

        let (q, k, v) = self.calcola_qkv(h, pos);
        k_cache.extend_from_slice(&k);
        v_cache.extend_from_slice(&v);

        let attn = self.attenzione_grezza(&q, k_cache, v_cache, pos + 1, 0, pos);
        self.uscita(h, &attn, out);
    }

    /// Prefill: `t` posizioni in un colpo (`h`/`out` flat `[t, n_embd]`).
    ///
    /// Ogni posizione vede solo il prefisso causale (e la finestra, se
    /// sliding). Le posizioni assolute partono da `pos0`. Nessuna cache
    /// persistente: è il riferimento CPU, non il percorso del motore.
    pub fn forward_seq(&self, h: &[f32], t: usize, pos0: usize, out: &mut [f32]) {
        let s = &self.spec;
        let ne = s.n_embd;
        let nkv = self.n_kv();
        assert_eq!(
            h.len(),
            t * ne,
            "attention seq: h ha {} elementi, attesi {t}*{ne}",
            h.len()
        );
        assert_eq!(
            out.len(),
            t * ne,
            "attention seq: out ha {} elementi, attesi {t}*{ne}",
            out.len()
        );
        if t == 0 {
            return;
        }

        // K/V dell'intero prefisso, in ordine di posizione.
        let mut ks = vec![0.0f32; t * nkv];
        let mut vs = vec![0.0f32; t * nkv];
        for p in 0..t {
            let hp = &h[p * ne..(p + 1) * ne];
            let (q, k, v) = self.calcola_qkv(hp, pos0 + p);
            ks[p * nkv..(p + 1) * nkv].copy_from_slice(&k);
            vs[p * nkv..(p + 1) * nkv].copy_from_slice(&v);

            let attn =
                self.attenzione_grezza(&q, &ks[..(p + 1) * nkv], &vs[..(p + 1) * nkv], p + 1, pos0, pos0 + p);
            self.uscita(hp, &attn, &mut out[p * ne..(p + 1) * ne]);
        }
    }
}

/// RMSNorm di una testa, in place: `x * rsqrt(mean(x²)+eps) * w`.
/// Stessa formula di `rmsnorm_row` in `qwen2.rs` (T5LayerNorm della reference).
fn rmsnorm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(w.len(), n, "rmsnorm: peso di {} elementi per testa di {n}", w.len());
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv * w[i];
    }
}

/// RoPE `rotate_half` sui gruppi di `head_dim`.
///
/// `inv_freq[i] = theta^(-2i/head_dim)` per `i < rope_angles`, zero oltre:
/// con inv_freq 0 l'angolo è 0, quindi cos=1 e sin=0 e le dimensioni
/// restano intatte (coda non ruotata del partial RoPE).
fn rope_rotante(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    rope_angles: usize,
    pos: usize,
) {
    let half = head_dim / 2;
    let n_ang = rope_angles.min(half);
    assert_eq!(
        x.len(),
        n_heads * head_dim,
        "rope: {} elementi per {n_heads} teste da {head_dim}",
        x.len()
    );
    for hh in 0..n_heads {
        let base = hh * head_dim;
        for i in 0..n_ang {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let ang = pos as f32 * freq;
            let (sin, cos) = ang.sin_cos();
            let x1 = x[base + i];
            let x2 = x[base + i + half];
            x[base + i] = x1 * cos - x2 * sin;
            x[base + i + half] = x2 * cos + x1 * sin;
        }
    }
}

/// Sigmoide stabile.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verità misurata sull'GGUF ufficiale LFM3-1.7B-F16 (27/10/2026).
    const KV_HEADS_REALI: [usize; 24] = [
        0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0,
    ];
    const SLIDING_REALI: [bool; 24] = [
        false, false, false, false, true, false, false, false, true, false, false, false, true, false,
        false, false, true, false, false, false, true, false, false, false,
    ];

    fn config_di_prova() -> Config {
        Config {
            n_layers: 24,
            n_embd: 2048,
            n_ff: 7680,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 10_000.0,
            rope_n_rot: 128,
            rms_eps: 1e-5,
            vocab_size: 128_000,
            sliding_window: 256,
            conv_l_cache: 7,
            kv_heads_per_layer: KV_HEADS_REALI.to_vec(),
            sliding_pattern: SLIDING_REALI.to_vec(),
        }
    }

    #[test]
    fn struttura_reale_13_5_6() {
        let c = config_di_prova();
        let (conv, swa, full) = c.kind_counts().unwrap();
        assert_eq!(conv, 13, "shortconv attesi 13");
        assert_eq!(swa, 5, "sliding attesi 5");
        assert_eq!(full, 6, "full attesi 6");
        assert_eq!(conv + swa + full, 24);
    }

    #[test]
    fn indici_attention_e_sliding_esatti() {
        let c = config_di_prova();
        let attn: Vec<usize> = (0..24)
            .filter(|&i| c.layer_kind(i).unwrap() != LayerKind::ShortConv)
            .collect();
        assert_eq!(attn, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
        let swa: Vec<usize> = (0..24)
            .filter(|&i| c.layer_kind(i).unwrap() == LayerKind::SlidingAttn)
            .collect();
        assert_eq!(swa, vec![4, 8, 12, 16, 20]);
    }

    #[test]
    fn due_rope_distinte_per_tipo_di_layer() {
        let c = config_di_prova();
        // layer 0 = shortconv: nessuna RoPE
        assert_eq!(c.rope_theta_for(0).unwrap(), None);
        // layer 2 = full attention: 1e6
        assert_eq!(c.rope_theta_for(2).unwrap(), Some(1_000_000.0));
        // layer 4 = sliding attention: 1e4  <-- se questo test cade, la parity salterà
        assert_eq!(c.rope_theta_for(4).unwrap(), Some(10_000.0));
    }

    #[test]
    fn layer_fuori_range_e_errore_esplicito() {
        let c = config_di_prova();
        let e = c.layer_kind(24).unwrap_err();
        assert!(e.contains("fuori range"), "errore poco chiaro: {e}");
    }

    #[test]
    fn parse_array_u32_e_u64() {
        let v = GgufValue::Array(vec![
            GgufValue::U32(0),
            GgufValue::U32(4),
            GgufValue::U64(7),
        ]);
        assert_eq!(parse_u_array(&v, "test").unwrap(), vec![0, 4, 7]);
    }

    #[test]
    fn parse_array_bool_con_interi_zero_uno() {
        let v = GgufValue::Array(vec![
            GgufValue::Bool(false),
            GgufValue::U32(1),
            GgufValue::Bool(true),
        ]);
        assert_eq!(
            parse_bool_array(&v, "test").unwrap(),
            vec![false, true, true]
        );
    }

    #[test]
    fn parse_array_rifiuta_scalare() {
        let v = GgufValue::U32(4);
        let e = parse_u_array(&v, "head_count_kv").unwrap_err();
        assert!(e.contains("atteso array"), "errore poco chiaro: {e}");
    }

    // ── F10-6: forward shortconv con dimensioni finte piccole ──
    // n_embd = 8 (invece di 2048) così i cicli si leggono a occhio;
    // l_cache = 7 come sul modello reale.
    const SC_NE: usize = 8;
    const SC_LCACHE: usize = 7;

    /// LCG numerico in [-1, 1): pseudocasuale fisso, niente dipendenze.
    fn lcg_next(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*s >> 33) as u32;
        (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn vec_lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg_next(&mut s)).collect()
    }

    fn block_di_prova() -> ShortConvBlock {
        ShortConvBlock::new(
            SC_NE,
            SC_LCACHE,
            vec_lcg(3 * SC_NE * SC_NE, 0xC0_FFEE),
            vec_lcg(SC_NE * SC_LCACHE, 0xB1_0CC0),
            vec_lcg(SC_NE * SC_NE, 0x0B_1EC7),
        )
        .expect("forme di prova corrette")
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn conv_contro_reference_naive() {
        let blk = block_di_prova();
        let ne = SC_NE;
        let kk = SC_LCACHE;
        let hr = kk - 1;
        let t = 5;
        let h = vec_lcg(t * ne, 12345);

        let mut out = vec![0.0f32; t * ne];
        let mut buf = vec![0.0f32; hr * ne];
        blk.forward_seq(&h, t, &mut buf, &mut out);

        // Reference scritta a mano nel test (cicli espliciti, nessuna
        // funzione condivisa col codice in prova) della formula
        // out_proj(C * conv_k7(B * x)).
        let mut ref_out = vec![0.0f32; t * ne];
        let mut hist = vec![0.0f32; (hr + t) * ne]; // zero-pad a sinistra
        for p in 0..t {
            let mut bcx = vec![0.0f32; 3 * ne];
            for j in 0..3 * ne {
                let mut s = 0.0f32;
                for i in 0..ne {
                    s += h[p * ne + i] * blk.in_proj[j * ne + i];
                }
                bcx[j] = s;
            }
            let mut g = vec![0.0f32; ne];
            for c in 0..ne {
                g[c] = bcx[c] * bcx[2 * ne + c];
            }
            hist[(hr + p) * ne..(hr + p + 1) * ne].copy_from_slice(&g);
            let mut cv = vec![0.0f32; ne];
            for c in 0..ne {
                let mut s = 0.0f32;
                for k2 in 0..kk {
                    s += blk.conv[c * kk + k2] * hist[(p + k2) * ne + c];
                }
                cv[c] = s;
            }
            for j in 0..ne {
                let mut s = 0.0f32;
                for i in 0..ne {
                    s += bcx[ne + i] * cv[i] * blk.out_proj[j * ne + i];
                }
                ref_out[p * ne + j] = s;
            }
        }

        let d = max_diff(&out, &ref_out);
        assert!(d <= 1e-5, "prefill vs reference naive: max diff {d:e} > 1e-5");

        // Anche lo step singolo (buf azzerato) deve accordarsi con la
        // posizione 0 della reference.
        let mut out1 = vec![0.0f32; ne];
        let mut buf1 = vec![0.0f32; hr * ne];
        blk.forward_step(&h[..ne], &mut buf1, &mut out1);
        let d1 = max_diff(&out1, &ref_out[..ne]);
        assert!(d1 <= 1e-5, "step vs reference naive: max diff {d1:e} > 1e-5");
    }

    #[test]
    fn incrementale_uguale_a_prefill() {
        let blk = block_di_prova();
        let ne = SC_NE;
        let hr = SC_LCACHE - 1;
        let t = 12;
        let h = vec_lcg(t * ne, 777);

        let mut out_seq = vec![0.0f32; t * ne];
        let mut buf_seq = vec![0.0f32; hr * ne];
        blk.forward_seq(&h, t, &mut buf_seq, &mut out_seq);

        let mut out_step = vec![0.0f32; t * ne];
        let mut buf_step = vec![0.0f32; hr * ne];
        for p in 0..t {
            let mut o = vec![0.0f32; ne];
            blk.forward_step(&h[p * ne..(p + 1) * ne], &mut buf_step, &mut o);
            out_step[p * ne..(p + 1) * ne].copy_from_slice(&o);
        }

        let d = max_diff(&out_seq, &out_step);
        assert!(d <= 1e-6, "prefill vs 12 step: max diff {d:e} > 1e-6");
        let db = max_diff(&buf_seq, &buf_step);
        assert!(db <= 1e-6, "buf finale prefill vs step: max diff {db:e} > 1e-6");
    }

    #[test]
    fn causalita() {
        let blk = block_di_prova();
        let ne = SC_NE;
        let hr = SC_LCACHE - 1;
        let t = 6;
        let h = vec_lcg(t * ne, 4242);

        let mut out_a = vec![0.0f32; t * ne];
        let mut buf_a = vec![0.0f32; hr * ne];
        blk.forward_seq(&h, t, &mut buf_a, &mut out_a);

        // Cambia SOLO l'ultimo token: le uscite precedenti non devono muoversi.
        let mut h2 = h.clone();
        for i in 0..ne {
            h2[(t - 1) * ne + i] += 10.0;
        }
        let mut out_b = vec![0.0f32; t * ne];
        let mut buf_b = vec![0.0f32; hr * ne];
        blk.forward_seq(&h2, t, &mut buf_b, &mut out_b);

        let d_pre = max_diff(&out_a[..(t - 1) * ne], &out_b[..(t - 1) * ne]);
        assert!(d_pre <= 1e-6, "causalità violata: le prime posizioni cambiano ({d_pre:e})");

        // Sanity: l'ultima uscita DEVE cambiare (altrimenti il test è vacuo).
        let d_last = max_diff(&out_a[(t - 1) * ne..], &out_b[(t - 1) * ne..]);
        assert!(d_last > 1e-3, "test vacuo: l'ultimo token non influenza la sua uscita");
    }

    #[test]
    fn stato_azzerato_riparte() {
        let blk = block_di_prova();
        let ne = SC_NE;
        let hr = SC_LCACHE - 1;
        let h3 = vec_lcg(3 * ne, 31337);
        let mut h5 = h3.clone();
        h5.extend(vec_lcg(2 * ne, 999));

        let mut out3 = vec![0.0f32; 3 * ne];
        let mut buf3 = vec![0.0f32; hr * ne];
        blk.forward_seq(&h3, 3, &mut buf3, &mut out3);

        let mut out5 = vec![0.0f32; 5 * ne];
        let mut buf5 = vec![0.0f32; hr * ne];
        blk.forward_seq(&h5, 5, &mut buf5, &mut out5);

        // Con buffer azzerato i primi 3 passi non dipendono dal futuro.
        let d = max_diff(&out3, &out5[..3 * ne]);
        assert!(d <= 1e-6, "stato non azzerato: i primi 3 passi cambiano ({d:e})");
    }

    // ── F10-7: forward attenzione con dimensioni finte piccole ──
    // n_embd = 8, 2 teste q da 4, 1 KV head da 4: GQA con 2 teste per KV.
    const AT_NE: usize = 8;
    const AT_HD: usize = 4;
    const AT_NH: usize = 2;
    const AT_NKV: usize = 1;

    fn spec_attn(theta: f32, finestra: Option<usize>, angoli: usize) -> AttentionSpec {
        AttentionSpec {
            n_embd: AT_NE,
            n_heads: AT_NH,
            n_kv_heads: AT_NKV,
            head_dim: AT_HD,
            rms_eps: 1e-5,
            rope_theta: theta,
            rope_angles: angoli,
            finestra,
        }
    }

    fn block_attn(spec: AttentionSpec) -> AttentionBlock {
        let ne = AT_NE;
        let nq = AT_NH * AT_HD;
        let nkv = AT_NKV * AT_HD;
        AttentionBlock::new(
            spec,
            AttentionDense {
                q: vec_lcg(ne * nq, 0xA77E_0001),
                k: vec_lcg(ne * nkv, 0xA77E_0002),
                v: vec_lcg(ne * nkv, 0xA77E_0003),
                output: vec_lcg(ne * nq, 0xA77E_0004),
                gate: vec_lcg(ne * AT_NH, 0xA77E_0005),
                q_norm: vec_lcg(AT_HD, 0xA77E_0006),
                k_norm: vec_lcg(AT_HD, 0xA77E_0007),
            },
        )
        .expect("forme di prova corrette")
    }

    #[test]
    fn forme_qkv_attese() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        assert_eq!(blk.n_q(), AT_NH * AT_HD);
        assert_eq!(blk.n_kv(), AT_NKV * AT_HD);
        let h = vec_lcg(AT_NE, 42);
        let (q, k, v) = blk.calcola_qkv(&h, 3);
        assert_eq!(q.len(), blk.n_q(), "q deve essere [n_heads*head_dim]");
        assert_eq!(k.len(), blk.n_kv(), "k deve essere [n_kv_heads*head_dim]");
        assert_eq!(v.len(), blk.n_kv(), "v deve essere [n_kv_heads*head_dim]");
        assert!(q.iter().chain(&k).chain(&v).all(|x| x.is_finite()));
    }

    #[test]
    fn una_posizione_restituisce_v() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let q = vec_lcg(blk.n_q(), 11);
        let k = vec_lcg(blk.n_kv(), 12);
        let v = vec_lcg(blk.n_kv(), 13);
        // Una sola chiave: softmax = 1, quindi l'uscita (pre-gate) è v,
        // replicato sulle due teste q che condividono la stessa KV head.
        let out = blk.attenzione_grezza(&q, &k, &v, 1, 0, 0);
        assert_eq!(out.len(), blk.n_q());
        for hh in 0..AT_NH {
            let d = max_diff(&out[hh * AT_HD..(hh + 1) * AT_HD], &v);
            assert!(d <= 1e-7, "testa {hh}: attenzione a una posizione != v ({d:e})");
        }
    }

    #[test]
    fn gate_a_zero_azzera_l_attenzione() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let mut attn = vec_lcg(blk.n_q(), 21);
        blk.applica_gate(&mut attn, &vec![0.0f32; AT_NH]);
        assert!(
            attn.iter().all(|&x| x == 0.0),
            "gate nullo non ha azzerato l'uscita: {attn:?}"
        );
    }

    #[test]
    fn maschera_sliding_esclude_i_vecchi() {
        // Finestra 2: la query in posizione 5 vede SOLO le chiavi 4 e 5
        // (riferimento HF: k_abs > q_abs - w), non la 3.
        let blk = block_attn(spec_attn(1_000_000.0, Some(2), AT_HD / 2));
        let t = 6;
        let hd = AT_HD;
        let q = vec![0.0f32; blk.n_q()]; // score tutti uguali → media delle v visibili
        let k = vec![0.0f32; t * blk.n_kv()];
        let mut v = vec![0.0f32; t * blk.n_kv()];
        for j in 0..t {
            for d in 0..hd {
                v[j * hd + d] = (j * 10 + d) as f32;
            }
        }
        let out = blk.attenzione_grezza(&q, &k, &v, t, 0, 5);
        // Uscita attesa: media di v[4] e v[5] (tutte e due le teste q).
        for hh in 0..AT_NH {
            for d in 0..hd {
                let attesa = ((4 * 10 + d) as f32 + (5 * 10 + d) as f32) / 2.0;
                let got = out[hh * hd + d];
                assert!(
                    (got - attesa).abs() <= 1e-5,
                    "finestra: testa {hh} dim {d}: {got} != {attesa} (chiave vecchia inclusa?)"
                );
            }
        }
    }

    #[test]
    fn finestra_globale_vede_tutto_il_prefisso() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let t = 6;
        let hd = AT_HD;
        let q = vec![0.0f32; blk.n_q()];
        let k = vec![0.0f32; t * blk.n_kv()];
        let mut v = vec![0.0f32; t * blk.n_kv()];
        for j in 0..t {
            for d in 0..hd {
                v[j * hd + d] = j as f32;
            }
        }
        let out = blk.attenzione_grezza(&q, &k, &v, t, 0, 5);
        let attesa = (0..t).map(|j| j as f32).sum::<f32>() / t as f32;
        for &x in &out {
            assert!((x - attesa).abs() <= 1e-5, "globale: {x} != {attesa}");
        }
    }

    #[test]
    fn due_theta_rotazioni_diverse() {
        let h = vec_lcg(AT_NE, 33);
        let piena = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let swa = block_attn(spec_attn(10_000.0, None, AT_HD / 2));
        // a pos 0 l'angolo è 0 per entrambe: q identico
        let (q0a, _, _) = piena.calcola_qkv(&h, 0);
        let (q0b, _, _) = swa.calcola_qkv(&h, 0);
        assert!(max_diff(&q0a, &q0b) <= 1e-7, "pos 0: theta diverse non devono contare");
        // a pos > 0 le rotazioni divergono
        let (qa, _, _) = piena.calcola_qkv(&h, 17);
        let (qb, _, _) = swa.calcola_qkv(&h, 17);
        assert!(
            max_diff(&qa, &qb) > 1e-4,
            "theta 1e6 e 1e4 devono ruotare diversamente (diff {:e})",
            max_diff(&qa, &qb)
        );
    }

    #[test]
    fn rope_parziale_non_ruota_la_coda() {
        let mut x = vec_lcg(AT_HD, 55);
        let prima = x.clone();
        // Un solo angolo su due: le dimensioni i=1 e i+half non ruotano.
        rope_rotante(&mut x, 1, AT_HD, 1_000.0, 1, 7);
        assert!((x[0] - prima[0]).abs() > 1e-4, "la prima dimensione deve ruotare");
        assert!((x[1] - prima[1]).abs() <= 1e-7, "la coda non deve ruotare");
        assert!((x[3] - prima[3]).abs() <= 1e-7, "la coda non deve ruotare");
    }

    #[test]
    fn step_uguale_a_prefill() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let t = 5;
        let h = vec_lcg(t * AT_NE, 66);

        let mut out_seq = vec![0.0f32; t * AT_NE];
        blk.forward_seq(&h, t, 0, &mut out_seq);

        let mut out_step = vec![0.0f32; t * AT_NE];
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for p in 0..t {
            let mut o = vec![0.0f32; AT_NE];
            blk.forward_step(&h[p * AT_NE..(p + 1) * AT_NE], p, &mut kc, &mut vc, &mut o);
            out_step[p * AT_NE..(p + 1) * AT_NE].copy_from_slice(&o);
        }

        let d = max_diff(&out_seq, &out_step);
        assert!(d <= 1e-6, "prefill vs step: max diff {d:e} > 1e-6");
    }

    #[test]
    fn causalita_attention() {
        let blk = block_attn(spec_attn(1_000_000.0, None, AT_HD / 2));
        let t = 5;
        let h = vec_lcg(t * AT_NE, 77);

        let mut out_a = vec![0.0f32; t * AT_NE];
        blk.forward_seq(&h, t, 0, &mut out_a);

        let mut h2 = h.clone();
        for i in 0..AT_NE {
            h2[(t - 1) * AT_NE + i] += 10.0;
        }
        let mut out_b = vec![0.0f32; t * AT_NE];
        blk.forward_seq(&h2, t, 0, &mut out_b);

        let d_pre = max_diff(&out_a[..(t - 1) * AT_NE], &out_b[..(t - 1) * AT_NE]);
        assert!(d_pre <= 1e-6, "causalità violata: le prime posizioni cambiano ({d_pre:e})");
        let d_last = max_diff(&out_a[(t - 1) * AT_NE..], &out_b[(t - 1) * AT_NE..]);
        assert!(d_last > 1e-4, "test vacuo: l'ultimo token non influenza la sua uscita");
    }
}
