//! Pre-tokenizzazione con regex (famiglia GPT-2 / Qwen2).
//!
//! La regex Qwen2 (da tokenizer.json di Qwen2.5, identica a quella usata
//! da llama.cpp per pre='qwen2'):
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//!    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!
//! Usa fancy-regex perché serve il lookahead negativo (?!\S).

use fancy_regex::Regex;
use super::TokenizerError;

#[derive(Debug, thiserror::Error)]
pub enum PreError {
    #[error("regex: {0}")]
    Regex(#[from] fancy_regex::Error),
    #[error("famiglia pre-tokenizer sconosciuta: {0}")]
    UnknownKind(String),
}

pub struct PreTokenizer {
    regex: Regex,
}

impl PreTokenizer {
    /// Regex per la famiglia indicata dal KV tokenizer.ggml.pre.
    /// 'qwen2', 'gpt2', 'llama3' e default cadono qui (llama3 ha regex propria,
    /// per ora approssimata con questa — la parity lo dirà).
    pub fn for_kind(kind: &str) -> std::result::Result<Self, TokenizerError> {
        let pattern = match kind {
            "qwen2" | "gpt2" | "default" => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            other => return Err(TokenizerError::UnknownPre(other.to_string())),
        };
        Ok(Self { regex: Regex::new(pattern).map_err(PreError::Regex)? })
    }

    /// Segmenta il testo in pre-token (non sovrapposti, in ordine).
    pub fn split(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut last = 0usize;
        // iteriamo sui match; i gap (caratteri non coperti) diventano pre-token essi stessi
        let mut matches = self.regex.find_iter(text);
        for m in matches.by_ref() {
            let Ok(m) = m else { break };
            let start = m.start();
            if start > last {
                out.push(text[last..start].to_string());
            }
            out.push(m.as_str().to_string());
            last = m.end();
        }
        if last < text.len() {
            out.push(text[last..].to_string());
        }
        out
    }
}

/// Mapping byte → carattere unicode (bytes_to_unicode di GPT-2, usato da
/// tutte le architetture byte-level BPE: gpt2, qwen2, ...).
pub fn bytes_to_unicode() -> Vec<char> {
    // range stampabili che GPT-2 mappa a se stessi
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);

    let mut map = vec![0u32; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        if bs.contains(&b) {
            map[b as usize] = b;
        } else {
            // i byte non stampabili → lettere latine consecutive da 256
            while bs.contains(&(256 + n)) {
                n += 1;
            }
            map[b as usize] = 256 + n;
            n += 1;
        }
    }
    map.into_iter().filter_map(char::from_u32).collect()
}
