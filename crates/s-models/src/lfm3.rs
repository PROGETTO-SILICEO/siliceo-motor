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
//! STATO: questo modulo copre SOLO config + classificazione dei layer + test.
//! Il forward NON c'è ancora, quindi `Famiglia::Lfm3` resta fuori da
//! `registry::Famiglia::from_arch` finché la parity non esiste — regola del
//! registro: "mai per speranza". Il server rifiuta `lfm3` con errore esplicito.

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
}
