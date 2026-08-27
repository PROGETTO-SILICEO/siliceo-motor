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
    /// Posizioni del prompt NON ricalcolate perché già presenti nella cache
    /// ricevuta come seme (prefix caching, AC-4). 0 = prefill completo.
    pub prefix_reused: usize,
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
    let mut nessuno = |_: &M::Cache| {};
    generate_tokens_seeded(
        model, prompt_ids, params, eos_all, decode, on_token, masker, None, &mut nessuno,
    )
}

/// Come `generate_tokens`, ma la generazione può PARTIRE da uno stato di
/// sequenza già riempito: è il meccanismo del prefix caching del server (AC-4).
///
/// `seed = Some((cache, n))` significa: i primi `n` token del prompt sono
/// GIÀ processati in `cache`. Il prefill ricalcola solo il delta `prompt[n..]`,
/// che per causalità produce logits IDENTICI al prefill completo — l'invariante
/// è verificato dai test (`seed == no-seed` al byte).
///
/// Vincoli del seme (scelti per correttezza, non ottimizzazione):
/// - `cache.len()` DEVE essere esattamente `n`: il delta viene scritto dalla
///   posizione corrente dello stato. È il chiamante — che conosce il tipo
///   concreto — ad allinearlo (`SeqState::truncate` per gli stati tutti-KV).
///   Un disallineamento scriverebbe il delta nella posizione sbagliata.
/// - `n` deve essere STRETTAMENTE < `prompt.len()`: se il prompt fosse
///   interamente coperto mancherebbero i logits dell'ultimo token. Per una
///   richiesta identica alla cached, il chiamante tronca il seme a len-1:
///   riprocessare UN token costa meno di un secondo percorso di codice.
/// - La CAPACITÀ dello stato non si può controllare qui (il tipo Cache è
///   opaco al trait): un seme troppo piccolo viene comunque respinto dal forward.
///
/// `on_prefill` è chiamata UNA volta, subito dopo il prefill completato:
/// il server la usa per clonare lo snapshot e metterlo in cache per le
/// richieste future (il prefisso di sistema degli agenti è enorme e ripetuto).
#[allow(clippy::too_many_arguments)]
pub fn generate_tokens_seeded<M: Device>(
    model: &M,
    prompt_ids: &[u32],
    params: &GenerateParams,
    eos_all: &[u32],
    decode: &dyn Fn(&[u32]) -> String,
    on_token: &mut dyn FnMut(u32) -> bool,
    masker: &mut dyn FnMut() -> Option<Vec<bool>>,
    seed: Option<(M::Cache, usize)>,
    on_prefill: &mut dyn FnMut(&M::Cache),
) -> Generated {
    let start = std::time::Instant::now();
    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p);
    let mut rng = match params.seed {
        Some(s) => Rng::new(s),
        None => Rng::from_time(),
    };

    // cache dimensionata con margine per i token generati
    let max_seq = prompt_ids.len() + params.max_tokens;

    // ── prefill ──
    // Con seme valido: delta-prefill da n (n < len del prompt, vedi sopra).
    // Senza: cache nuova e prompt intero. Il resto del loop NON cambia:
    // un solo percorso di generazione per entrambi i casi.
    let mut prefix_reused = 0usize;
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let mut stopped_text: Option<String> = None;
    let mut last_token;
    // OpenAI semantics: "stop" se fermati da EOS o stop-string, "length" solo
    // se esauriti i token senza fermata naturale.
    let mut motivo = "length";

    let (mut cache, mut next_logits);
    // seme ammissibile: prefisso VERO del prompt (n < len, rigoroso), non vuoto.
    // Il chiamante garantisce cache.len()==n; qui non è verificabile (Cache opaco).
    let usable = match seed {
        Some((c, n)) if !prompt_ids.is_empty() && n > 0 && n < prompt_ids.len() => Some((c, n)),
        _ => None,
    };
    if let Some((mut c, n)) = usable {
        // il delta include l'ultimo token del prompt → forward_cached ritorna
        // direttamente i suoi logits, come nel prefill pieno
        next_logits = model.forward_cached(&prompt_ids[n..], &mut c);
        on_prefill(&c);
        prefix_reused = n;
        last_token = *prompt_ids.last().unwrap();
        cache = c;
    } else {
        cache = model.new_cache(max_seq);
        next_logits = model.forward_cached(prompt_ids, &mut cache);
        on_prefill(&cache);
        last_token = *prompt_ids.last().unwrap_or(&0);
    }

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
        prefix_reused,
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

    #[derive(Clone)]
    struct FakeCache {
        len: usize,
        max_seq: usize,
        /// numero di chiamate a forward_cached (= passi di generazione fatti)
        calls: usize,
    }

    impl FakeCache {
        /// Speculare a SeqState::truncate: il chiamante allinea il seme.
        fn truncate(&mut self, n: usize) {
            assert!(n <= self.len);
            self.len = n;
        }
    }

    impl Device for FakeDevice {
        type Cache = FakeCache;
        type Config = Config;

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
            // logits FUNZIONE DELLA POSIZIONE (cache.len-1), non del numero di
            // chiamate: è il contratto Device reso esplicito. Un fake basato
            // sui contatori di chiamata sembrerebbe cambiare stati con lo
            // stesso contenuto → falsificherebbe il prefix caching (AC-4).
            let p = cache.len - 1;
            let winner = self.sequence.get(p).copied().unwrap_or(7);
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
        // prefill (pos 1 → winner sequence[1]=9) + decode: poi 15, infine il
        // token di fallback (7) oltre la sequenza, fino a max_tokens → "length"
        assert_eq!(out.ids.len(), 10);
        assert_eq!(&out.ids[..3], &[9, 15, 7]);
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

    // ── AC-4 prefix caching ──

    /// Genera con seme None e cattura lo snapshot post-prefill.
    fn prima_passata(
        dev: &FakeDevice,
        prompt: &[u32],
        max_tokens: usize,
    ) -> (Generated, (FakeCache, usize)) {
        let params = GenerateParams { max_tokens, ..Default::default() };
        let mut tutti = |_: u32| -> bool { true };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let snap: std::cell::RefCell<Option<(FakeCache, usize)>> =
            std::cell::RefCell::new(None);
        let mut on_prefill = |c: &FakeCache| {
            *snap.borrow_mut() = Some((c.clone(), prompt.len()));
        };
        let out = generate_tokens_seeded(
            dev,
            prompt,
            &params,
            &[],
            &|ids| ids.iter().map(|i| i.to_string()).collect(),
            &mut tutti,
            &mut nomask,
            None,
            &mut on_prefill,
        );
        let seed = snap.into_inner().unwrap();
        (out, seed)
    }

    #[test]
    fn seeded_uguale_al_byte_al_percorso_semplice() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3, 9, 15] };

        // 1a richiesta: prefill pieno (max_tokens alto → snapshot capiente
        // anche per la richiesta ESTESA più sotto: la capacità è contratto
        // del chiamante, come documentato su generate_tokens_seeded)
        let (base, (snapshot, n)) = prima_passata(&dev, &[1, 2], 20);
        assert_eq!(n, 2);
        assert_eq!(base.prefix_reused, 0);

        // 2a richiesta IDENTICA: il seme copre tutto il prompt → chi semina
        // lo tronca a len-1 (il chiamante allinea cache.len()==n, vedi contratto)
        let mut seme = snapshot.clone();
        seme.truncate(n - 1);
        let params = GenerateParams { max_tokens: 6, ..Default::default() };
        let mut tutti = |_: u32| -> bool { true };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let mut nessuno = |_: &FakeCache| {};
        let out2 = generate_tokens_seeded(
            &dev,
            &[1, 2],
            &params,
            &[],
            &|ids| ids.iter().map(|i| i.to_string()).collect(),
            &mut tutti,
            &mut nomask,
            Some((seme, n - 1)),
            &mut nessuno,
        );
        assert_eq!(out2.prefix_reused, 1); // clamppato lato chiamante
        // stesso output al byte col seme (base genera di più: confronto i primi)
        assert_eq!(out2.ids, base.ids[..out2.ids.len()]);

        // 3a richiesta che ESTENDE il prefisso (tipico agente: stessa storia + nuovo turno).
        // Verifica forte: seeded su prompt esteso == no-seed SULLO STESSO prompt esteso.
        let mut esteso = vec![1u32, 2];
        esteso.extend_from_slice(&[5, 6, 7]);
        let mut nessuno2 = |_: &FakeCache| {};
        let out3 = generate_tokens_seeded(
            &dev,
            &esteso,
            &params,
            &[],
            &|ids| ids.iter().map(|i| i.to_string()).collect(),
            &mut tutti,
            &mut nomask,
            Some((snapshot.clone(), n)),
            &mut nessuno2,
        );
        assert_eq!(out3.prefix_reused, 2, "prefisso intero riusato");
        let out3_pieno = generate_tokens_seeded(
            &dev,
            &esteso,
            &params,
            &[],
            &|ids| ids.iter().map(|i| i.to_string()).collect(),
            &mut tutti,
            &mut nomask,
            None,
            &mut nessuno2,
        );
        assert_eq!(out3.ids, out3_pieno.ids, "seeded == pieno anche sul prompt esteso");
    }

    #[test]
    fn seme_inutilizzabile_cade_sul_prefill_pieno() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3, 9, 15] };
        let (_, (snapshot, _)) = prima_passata(&dev, &[1, 2], 4);

        // n=0 → inutile; n > len → bug del chiamante → scartato
        for bad in [0usize, 5] {
            let params = GenerateParams { max_tokens: 4, ..Default::default() };
            let mut tutti = |_: u32| -> bool { true };
            let mut nomask = || -> Option<Vec<bool>> { None };
            let mut nessuno = |_: &FakeCache| {};
            let out = generate_tokens_seeded(
                &dev,
                &[1, 2],
                &params,
                &[],
                &|ids| ids.iter().map(|i| i.to_string()).collect(),
                &mut tutti,
                &mut nomask,
                Some((snapshot.clone(), bad)),
                &mut nessuno,
            );
            assert_eq!(out.prefix_reused, 0);
        }
    }

    #[test]
    fn on_prefill_arriva_una_sola_volta_con_lo_stato_lungo_quanto_il_prompt() {
        let dev = FakeDevice { config: fake_config(), sequence: vec![3] };
        let params = GenerateParams { max_tokens: 2, ..Default::default() };
        let mut tutti = |_: u32| -> bool { true };
        let mut nomask = || -> Option<Vec<bool>> { None };
        let chiamate = std::cell::Cell::new(0);
        let len_vista = std::cell::Cell::new(0usize);
        let mut on_prefill = |c: &FakeCache| {
            chiamate.set(chiamate.get() + 1);
            len_vista.set(c.len);
        };
        generate_tokens_seeded(
            &dev,
            &[1, 2, 3],
            &params,
            &[],
            &|ids| ids.iter().map(|i| i.to_string()).collect(),
            &mut tutti,
            &mut nomask,
            None,
            &mut on_prefill,
        );
        assert_eq!(chiamate.get(), 1);
        assert_eq!(len_vista.get(), 3, "lo snapshot copre ESATTAMENTE il prompt");
    }
}
