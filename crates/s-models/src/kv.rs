//! KV cache incrementale — F2.5.
//!
//! Il forward full-recompute di F1 costa O(n²) sulla lunghezza della sequenza:
//! ad ogni nuovo token ricalcola attenzione e proiezioni per TUTTI i token
//! precedenti. Con la cache, K e V dei token già processati vengono riusati:
//! il decode diventa O(n) — un solo token in input per passo.
//!
//! Invariante di correttezza: `forward_cached(prefill) + forward_cached(decode)`
//! deve produrre logits IDENTICI al full-recompute. Verificato dai test.

use crate::qwen2::Config;

/// Cache K/V di un singolo layer: [pos × kv_dim] ciascuna.
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Cache multi-layer per una sequenza.
pub struct KvCache {
    layers: Vec<LayerKv>,
    kv_dim: usize,
    /// Numero di posizioni già scritte (lunghezza della sequenza processata).
    pub len: usize,
}

impl KvCache {
    pub fn new(config: &Config, max_seq: usize) -> Self {
        let kv_dim = config.n_kv_heads * config.head_dim;
        Self {
            layers: (0..config.n_layers)
                .map(|_| LayerKv {
                    k: vec![0.0; max_seq * kv_dim],
                    v: vec![0.0; max_seq * kv_dim],
                })
                .collect(),
            kv_dim,
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

// Accesso interno per il forward (i campi restano privati fuori dal crate).
impl KvCache {
    /// Slice K del layer da posizione `n_positions * kv_dim` in poi (zona scrittura).
    pub(crate) fn layer_k_mut(&mut self, li: usize) -> &mut [f32] {
        let start = self.len * self.kv_dim;
        &mut self.layers[li].k[start..]
    }
    pub(crate) fn layer_v_mut(&mut self, li: usize) -> &mut [f32] {
        let start = self.len * self.kv_dim;
        &mut self.layers[li].v[start..]
    }
    /// Slice K del layer per le prime `n_positions` posizioni (zona lettura).
    pub(crate) fn layer_k_upto(&self, li: usize, n_positions: usize) -> &[f32] {
        &self.layers[li].k[..n_positions * self.kv_dim]
    }
    pub(crate) fn layer_v_upto(&self, li: usize, n_positions: usize) -> &[f32] {
        &self.layers[li].v[..n_positions * self.kv_dim]
    }

    /// Avanza la lunghezza dopo aver scritto n nuove posizioni.
    pub(crate) fn advance(&mut self, n: usize) {
        self.len += n;
    }
}
