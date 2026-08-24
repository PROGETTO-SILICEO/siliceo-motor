//! s-tokenizer — BPE byte-level nativa per siliceo-motor.
//!
//! Legge vocabolario e merges DIRETTAMENTE dal GGUF (tokenizer.ggml.tokens,
//! tokenizer.ggml.merges): zero dipendenze esterne, D2 risolta in senso sovrano.
//!
//! Pre-tokenizzazione: regex Qwen2 (identica a quella di HF transformers per
//! architetture qwen2/llama con pre 'qwen2'/'gpt2' family).

pub mod bpe;
pub mod pre;

use std::collections::HashMap;
use std::path::Path;

use s_gguf::GgufValue;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("GGUF: {0}")]
    Gguf(#[from] s_gguf::GgufError),
    #[error("KV mancante nel GGUF: {0}")]
    MissingKv(String),
    #[error("tipo KV inatteso per {0}")]
    BadKvType(String),
    #[error("token non trovato nel vocabolario: {0:?}")]
    UnknownToken(String),
    #[error("pre-tokenizer sconosciuto: {0}")]
    UnknownPre(String),
    #[error("pre-tokenizzazione: {0}")]
    Pre(#[from] pre::PreError),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Tokenizer BPE byte-level caricato da un file GGUF.
pub struct Tokenizer {
    /// Vocabolario indicizzato per id.
    vocab: Vec<String>,
    /// id → stringa, per lookup merges e encoding.
    token_to_id: HashMap<String, u32>,
    /// Ranks dei merges: coppia (left, right) → priorità (più bassa = prima).
    merge_ranks: HashMap<(u32, u32), u32>,
    /// Pre-tokenizzatore (regex per famiglia 'qwen2'/'gpt2').
    pre: pre::PreTokenizer,
    /// byte → carattere unicode (mapping GPT-2 byte-level).
    byte_encoder: Vec<char>,
    /// carattere unicode → byte.
    byte_decoder: HashMap<char, u8>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
}

impl Tokenizer {
    /// Carica il tokenizer da un file GGUF (usa s-gguf per la lettura).
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = s_gguf::GgufFile::open(path)?;

        let tokens = gguf_array_str(&gguf, "tokenizer.ggml.tokens")?;
        let merges = gguf_array_str(&gguf, "tokenizer.ggml.merges")?;
        let token_types = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.token_type")
            .ok_or_else(|| TokenizerError::MissingKv("tokenizer.ggml.token_type".into()))?;
        let _ = token_types; // per ora non filtriamo: byte fallback gestito dal vocabolario

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        // merges: ogni entry è "left right" — risolviamo subito in coppie di ID
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m
                .split_once(' ')
                .ok_or_else(|| TokenizerError::BadKvType(format!("merge malformato: {m:?}")))?;
            if let (Some(&ia), Some(&ib)) = (token_to_id.get(a), token_to_id.get(b)) {
                merge_ranks.insert((ia, ib), rank as u32);
            }
            // coppie non risolvibili: merge morto nel file, lo saltiamo
        }

        let byte_encoder = pre::bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder
            .iter()
            .enumerate()
            .map(|(byte, &ch)| (ch, byte as u8))
            .collect();

        let bos_id = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.bos_token_id")
            .and_then(|(_, v)| v.as_u64())
            .map(|v| v as u32);
        let eos_id = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.eos_token_id")
            .and_then(|(_, v)| v.as_u64())
            .map(|v| v as u32);
        let add_bos = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.add_bos_token")
            .map(|(_, v)| matches!(v, GgufValue::Bool(true)))
            .unwrap_or(false);

        let pre_kind = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.pre")
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("qwen2")
            .to_string();

        Ok(Self {
            vocab: tokens,
            token_to_id,
            merge_ranks,
            pre: pre::PreTokenizer::for_kind(&pre_kind)?,
            byte_encoder,
            byte_decoder,
            bos_id,
            eos_id,
            add_bos,
        })
    }

    /// Codifica testo → token IDs (senza BOS; usa `encode_with_specials` per il comportamento completo).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if self.add_bos {
            if let Some(b) = self.bos_id {
                ids.push(b);
            }
        }
        for piece in self.pre.split(text) {
            ids.extend(self.encode_piece(&piece)?);
        }
        Ok(ids)
    }

    /// Codifica un singolo pre-token (già segmentato dalla regex).
    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>> {
        // byte-level: ogni byte → carattere unicode mappato (GPT-2 style)
        let mapped: String = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[b as usize])
            .collect();

        // parti in singoli caratteri e applica i merges in ordine di rank
        let mut parts: Vec<u32> = Vec::with_capacity(mapped.chars().count());
        for ch in mapped.chars() {
            let s = ch.to_string();
            let id = *self
                .token_to_id
                .get(&s)
                .ok_or_else(|| TokenizerError::UnknownToken(s.clone()))?;
            parts.push(id);
        }

        loop {
            // trova la coppia con rank più basso
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_ranks.get(&(parts[i], parts[i + 1])) {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let (_, i) = match best {
                Some(b) => b,
                None => break,
            };
            // fondi parts[i] e parts[i+1]
            let merged = format!("{}{}", self.vocab[parts[i] as usize], self.vocab[parts[i + 1] as usize]);
            let id = *self
                .token_to_id
                .get(&merged)
                .ok_or_else(|| TokenizerError::UnknownToken(merged.clone()))?;
            parts[i] = id;
            parts.remove(i + 1);
        }
        Ok(parts)
    }

    /// Decodifica IDs → testo (byte-level inverso).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self
                .vocab
                .get(id as usize)
                .ok_or(TokenizerError::UnknownToken(format!("<id {id}>")))?;
            // i token speciali (<|...|>) non hanno byte: li saltiamo in decodifica
            if tok.starts_with("<|") && tok.ends_with("|>") {
                continue;
            }
            for ch in tok.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        // carattere fuori dal mapping byte-level: token speciale
                        // o vocabolario esteso — lo aggiungiamo come UTF-8 diretto
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::UnknownToken("decode utf8".into()))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
    /// Lookup diretto id di un token esatto (es. token speciali "<|im_end|>").
    pub fn token_to_id(&self, tok: &str) -> Option<u32> {
        self.token_to_id.get(tok).copied()
    }
}

fn gguf_array_str(gguf: &s_gguf::GgufFile, key: &str) -> Result<Vec<String>> {
    let (_, val) = gguf
        .metadata
        .iter()
        .find(|(k, _)| k == key)
        .ok_or_else(|| TokenizerError::MissingKv(key.to_string()))?;
    match val {
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    GgufValue::String(s) => out.push(s.clone()),
                    _ => return Err(TokenizerError::BadKvType(key.to_string())),
                }
            }
            Ok(out)
        }
        _ => Err(TokenizerError::BadKvType(key.to_string())),
    }
}
