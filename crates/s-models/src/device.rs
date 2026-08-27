//! Astrazione del dispositivo di calcolo — F3a.
//!
//! Il loop di generazione (generate.rs) e il resto dello stack (server,
//! sampling, chat) devono essere IDENTICI su CPU e GPU: cambia solo chi
//! esegue il forward e dove vive la KV cache (RAM host vs memoria device).
//!
//! Il trait è qui in s-models (non in s-backend) perché Config e i modelli
//! concreti vivono già in questo crate: s-backend-cuda dipende da s-models
//! e implementa `Device` per il suo modello residente su GPU. Nessuna
//! dipendenza circolare, nessun tratto spostabile che non serve.
//!
//! Contratto di correttezza (stesso invariante dei test F2.5): per ogni
//! implementazione, `forward_cached(prefill) + forward_cached(decode)`
//! deve dare logits compatibili con il full-recompute della stessa
//! architettura. La parity cross-backend si verifica con lo stesso test
//! greedy multi-token usato per F1/F2.

/// Un backend di forward: CPU oggi, CUDA domani, Vulkan dopo.
/// F6: generalizzato per famiglie multiple (qwen2, qwen35) via Config associato.
pub trait Device {
    /// Cache specifica del backend (host Vec per CPU, buffer device per GPU).
    type Cache;
    type Config;

    fn config(&self) -> &Self::Config;

    /// Cache dimensionata per una sequenza massima `max_seq`.
    fn new_cache(&self, max_seq: usize) -> Self::Cache;

    /// Forward incrementale: processa `tokens` a partire dalla posizione
    /// assoluta `cache.len()` (prefill = tutto il prompt; decode = 1 token).
    /// Ritorna i logits dell'ULTIMO token [vocab_size].
    fn forward_cached(&self, tokens: &[u32], cache: &mut Self::Cache) -> Vec<f32>;
}
