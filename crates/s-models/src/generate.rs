//! Loop di generazione autoregressiva — F2.
//!
//! Responsabilità:
//! - far girare forward + sampling token per token
//! - stop conditions: max_tokens, EOS (multipli), stop strings
//! - decodifica del testo generato
//!
//! Nota performance: per F2 il forward è full-recompute (niente KV cache,
//! che arriva con s-kv). Correttezza prima della velocità.

use crate::sampling::{Rng, Sampler};
use crate::Model;
use s_tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Numero massimo di token DA GENERARE (escluso il prompt).
    pub max_tokens: usize,
    /// Temperatura (0 = greedy).
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Seed del RNG (None = seed dal tempo).
    pub seed: Option<u64>,
    /// Sequenze che interrompono la generazione quando compaiono nel testo.
    pub stop: Vec<String>,
    /// Token EOS aggiuntivi oltre a quelli del tokenizer (es. <|im_end|> in chat).
    pub extra_eos: Vec<u32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            stop: Vec::new(),
            extra_eos: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Generated {
    /// Token generati (prompt escluso). Possono includere token oltre
    /// l'ultimo punto di stop-string: il testo ritornato è già ritagliato.
    pub ids: Vec<u32>,
    /// Testo decodificato e ritagliato alle stop conditions.
    pub text: String,
    /// "stop" (EOS o stop string) oppure "length" (max_tokens raggiunto).
    pub finish_reason: &'static str,
    pub prompt_tokens: usize,
    /// Tempo totale di generazione.
    pub elapsed_secs: f32,
}

impl Generated {
    pub fn tokens_per_sec(&self) -> f32 {
        if self.elapsed_secs > 0.0 {
            self.ids.len() as f32 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// Genera da un prompt già tokenizzato.
///
/// Usa la KV cache incrementale: prefill una volta sola, poi un token per
/// passo. L'output è identico al full-recompute (verificato dai test).
pub fn generate(
    model: &Model,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
) -> Generated {
    let start = std::time::Instant::now();
    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p);
    let mut rng = match params.seed {
        Some(s) => Rng::new(s),
        None => Rng::from_time(),
    };

    // Insieme degli EOS: eos del tokenizer + extra (es. <|im_end|>)
    let eos_all: Vec<u32> =
        tok.eos_id().into_iter().chain(params.extra_eos.iter().copied()).collect();

    // cache dimensionata con margine per i token generati
    let max_seq = prompt_ids.len() + params.max_tokens;
    let mut cache = crate::kv::KvCache::new(&model.config, max_seq);

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let mut stopped_text: Option<String> = None;
    let mut next_logits;
    // OpenAI semantics: "stop" se fermati da EOS o stop-string, "length" solo
    // se esauriti i token senza fermata naturale.
    let mut motivo = "length";

    // ── prefill: tutto il prompt in un colpo ──
    next_logits = model.forward_cached(prompt_ids, &mut cache);
    let mut last_token = *prompt_ids.last().unwrap_or(&0);

    for step in 0..params.max_tokens {
        // il token appena predetto diventa l'input del prossimo passo
        if step > 0 {
            next_logits = model.forward_cached(&[last_token], &mut cache);
        }
        let next = sampler.sample(&next_logits, &mut rng) as u32;
        generated.push(next);
        last_token = next;

        if eos_all.contains(&next) {
            motivo = "stop";
            break;
        }

        // Stop strings: controlliamo sul testo decodificato fin qui.
        if !params.stop.is_empty() {
            let full = tok.decode(&generated).unwrap_or_default();
            if let Some(hit) = params.stop.iter().find(|s| !s.is_empty() && full.contains(s.as_str())) {
                let pos = full.find(hit.as_str()).unwrap_or(full.len());
                stopped_text = Some(full[..pos].to_string());
                motivo = "stop";
                break;
            }
        }
    }

    let finish_reason = motivo;
    let text = stopped_text.unwrap_or_else(|| tok.decode(&generated).unwrap_or_default());
    Generated {
        text,
        ids: generated,
        finish_reason,
        prompt_tokens: prompt_ids.len(),
        elapsed_secs: start.elapsed().as_secs_f32(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Nota: i test di integrazione del loop richiedono un modello reale.
    // Quelli del sampler sono in sampling.rs; qui testiamo solo la logica
    // dei default.
    #[test]
    fn defaults_sensati() {
        let p = GenerateParams::default();
        assert_eq!(p.temperature, 0.0); // greedy di default: deterministico
        assert_eq!(p.max_tokens, 256);
        assert!(p.stop.is_empty());
    }
}
