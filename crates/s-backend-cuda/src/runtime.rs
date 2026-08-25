//! Runtime CUDA (feature `cuda`) — scheletro F3b.
//!
//! Responsabilità previste, in ordine di implementazione:
//! 1. `CudaCtx`: init driver, selezione device, stream, gestione errori
//!    (ogni errore cust → messaggio con contesto, mai panic in serving).
//! 2. Pesi residenti su device: upload una tantum al load, zero copie
//!    host↔device nel decode (solo token ID in ingresso, logits in uscita).
//! 3. KV cache su device: stessa semantica di s-models::kv (posizioni
//!    assolute, append), buffer pre-allocati a max_seq.
//! 4. Kernels: matmul (prima f32/f16 naive poi tiled), rmsnorm, RoPE NEOX,
//!    softmax causale fused per-head, silu-mul, add residuo.
//!
//! Vincolo di correttezza: ogni kernel ha un test di parity contro la CPU
//! (stessi input → tolleranza documentata). La generazione greedy multi-token
//! end-to-end è il test definitivo, come insegnato da F1.

// Modulo attivato solo con --features cuda. Il corpo arriva con il primo
// accesso reale all'hardware (centro-operativo / RTX 3090): scrivere codice
// GPU non compilabile e non testabile viola le regole del progetto.
