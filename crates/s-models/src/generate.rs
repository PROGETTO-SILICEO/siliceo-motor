//! Loop di generazione autoregressiva — F2.
//!
//! Responsabilità:
//! - far girare forward + sampling token per token
//! - stop conditions: max_tokens, EOS (multipli), stop strings
//! - decodifica del testo generato
//!
//! Nota performance: per F2 il forward è full-recompute (niente KV cache,
//! che arriva con s-kv). Correttezza prima della velocità.

use crate::device::Device;
use crate::sampling::{Rng, Sampler};
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

/// Genera da un prompt già tokenizzato, su QUALSIASI backend (F3a).
///
/// Usa la KV cache incrementale del backend: prefill una volta sola, poi un
/// token per passo. Su CPU l'output è identico al full-recompute (verificato);
/// ogni nuovo backend deve passare lo stesso test greedy multi-token.
pub fn generate<M: Device>(
    model: &M,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
) -> Generated {
    let eos_all: Vec<u32> =
        tok.eos_id().into_iter().chain(params.extra_eos.iter().copied()).collect();
    let mut nessuno = |_: u32| -> bool { true };
    let mut nomask = || -> Option<Vec<bool>> { None };
    generate_tokens(
        model,
        prompt_ids,
        params,
        &eos_all,
        &|ids: &[u32]| tok.decode(ids).unwrap_or_default(),
        &mut nessuno,
        &mut nomask,
    )
}

/// Nucleo del loop, indipendente dal Tokenizer: riceve l'insieme EOS e una
/// funzione di decodifica. Serve ai test con backend finti e a F3 (la
/// decodifica resta host-side qualunque sia il device).
///
///
/// `on_token`: chiamata per OGNI token generato (dopo il sampling, prima delle
/// stop conditions). Ritorna false per interrompere la generazione (abort del
/// consumer, es. client SSE disconnesso): `finish_reason` resta quello raggiunto.
/// È il punto di aggancio dello streaming agentic (AC-1 in docs/AGENTIC_CORE.md):
/// UN SOLO loop interno, niente duplicazione del percorso verificato.
/// `masker`: chiamata PRIMA di ogni sampling; Some(mask) vincola i token
/// ammessi (AC-3 structured output), None = passo libero.
pub fn generate_tokens<M: Device>(
    model: &M,
    prompt_ids: &[u32],
    params: &GenerateParams,
    eos_all: &[u32],
    decode: &dyn Fn(&[u32]) -> String,
    on_token: &mut dyn FnMut(u32) -> bool,
    masker: &mut dyn FnMut() -> Option<Vec<bool>>,
) -> Generated {
    let start = std::time::Instant::now();
    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p);
    let mut rng = match params.seed {
        Some(s) => Rng::new(s),
        None => Rng::from_time(),
    };

    // cache dimensionata con margine per i token generati
    let max_seq = prompt_ids.len() + params.max_tokens;
    let mut cache = model.new_cache(max_seq);

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
        let mask = masker();
        let next = match &mask {
            Some(m) => sampler.sample_masked(&next_logits, m, &mut rng) as u32,
            None => sampler.sample(&next_logits, &mut rng) as u32,
        };
        generated.push(next);
        last_token = next;

        // streaming: il consumer può fermare la generazione (false = abort)
        if !on_token(next) {
            motivo = "stop";
            break;
        }

        if eos_all.contains(&next) {
            motivo = "stop";
            break;
        }

        // Stop strings: controlliamo sul testo decodificato fin qui.
        if !params.stop.is_empty() {
            let full = decode(&generated);
            if let Some(hit) = params.stop.iter().find(|s| !s.is_empty() && full.contains(s.as_str())) {
                let pos = full.find(hit.as_str()).unwrap_or(full.len());
                stopped_text = Some(full[..pos].to_string());
                motivo = "stop";
                break;
            }
        }
    }

    let finish_reason = motivo;
    let text = stopped_text.unwrap_or_else(|| decode(&generated));
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
    use crate::device::Device;
    use crate::qwen2::Config;

    // Nota: i test di integrazione del loop richiedono un modello reale.
    // Quelli del sampler sono in sampling.rs; qui testiamo la logica dei
    // default e il loop GENERICO con un backend finto (indipendente dal
    // modello: serve a F3 per garantire che generate non dipenda da CPU).

    #[test]
    fn defaults_sensati() {
        let p = GenerateParams::default();
        assert_eq!(p.temperature, 0.0); // greedy di default: deterministico
        assert_eq!(p.max_tokens, 256);
        assert!(p.stop.is_empty());
    }

    /// Backend finto: logits deterministici che ruotano sul vocabolario.
    /// Verifica che il loop generico (prefill → decode → EOS) funzioni
    /// attraverso il trait, senza toccare un vero modello.
    struct FakeDevice {
        config: Config,
        /// token emesso al passo n della generazione
        sequence: Vec<u32>,
    }

    struct FakeCache {
        len: usize,
        max_seq: usize,
        /// numero di chiamate a forward_cached (= passi di generazione fatti)
        calls: usize,
    }

    impl Device for FakeDevice {
        type Cache = FakeCache;

        fn config(&self) -> &Config {
            &self.config
        }

        fn new_cache(&self, max_seq: usize) -> FakeCache {
            FakeCache { len: 0, max_seq, calls: 0 }
        }

        fn forward_cached(&self, tokens: &[u32], cache: &mut FakeCache) -> Vec<f32> {
            assert!(!tokens.is_empty());
            assert!(cache.len + tokens.len() <= cache.max_seq);
            cache.calls += 1;
            cache.len += tokens.len();
            // logits: argmax = sequence[call-1] se esiste, altrimenti fallback
            let winner = self.sequence.get(cache.calls - 1).copied().unwrap_or(7);
            (0..self.config.vocab_size)
                .map(|i| if i as u32 == winner { 10.0 } else { 0.0 })
                .collect()
        }
    }

    fn fake_config() -> Config {
        Config {
            n_layers: 1,
            n_embd: 8,
            n_ff: 8,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: 16,
        }
    }

    #[test]
    fn loop_generico_attraverso_il_trait() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3, 9, 15] };
        let params = GenerateParams { max_tokens: 10, ..Default::default() };
        let mut tutti = |_: u32| -> bool { true };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let out = generate_tokens(&dev, &[1, 2], &params, &[], &|ids| {
            ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        }, &mut tutti, &mut nomask);
        // prefill + decode: sequenza 3, 9, 15; poi la FakeDevice ripete il
        // token di fallback (7) fino a max_tokens → finish_reason "length"
        assert_eq!(out.ids.len(), 10);
        assert_eq!(&out.ids[..3], &[3, 9, 15]);
        assert_eq!(out.finish_reason, "length");
        assert_eq!(out.prompt_tokens, 2);
    }

    #[test]
    fn loop_generico_eos_attraverso_il_trait() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3, 9] };
        let params = GenerateParams { max_tokens: 10, ..Default::default() };
        let mut tutti = |_: u32| -> bool { true };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let out = generate_tokens(&dev, &[1], &params, &[9], &|ids| {
            ids.iter().map(|i| i.to_string()).collect()
        }, &mut tutti, &mut nomask);
        // 3 generato, poi 9 == EOS → stop
        assert_eq!(out.ids, vec![3, 9]);
        assert_eq!(out.finish_reason, "stop");
    }

    /// AC-1: la callback può interrompere la generazione (abort consumer).
    #[test]
    fn callback_false_interrompe_la_generazione() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3, 9, 15] };
        let params = GenerateParams { max_tokens: 10, ..Default::default() };
        let visti = std::cell::RefCell::new(Vec::new());
        let mut aborti_dopo_due = |t: u32| -> bool {
            visti.borrow_mut().push(t);
            visti.borrow().len() < 2
        };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let out = generate_tokens(&dev, &[1], &params, &[], &|ids| {
            ids.iter().map(|i| i.to_string()).collect()
        }, &mut aborti_dopo_due, &mut nomask);
        assert_eq!(out.ids.len(), 2); // fermata al secondo token, non a max_tokens
        assert_eq!(*visti.borrow(), vec![3, 9]);
        assert_eq!(out.finish_reason, "stop");
    }
}
