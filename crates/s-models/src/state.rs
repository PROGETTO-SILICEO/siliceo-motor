//! Stato di sequenza generalizzato — F3b finestra 1.
//!
//! FINO A OGGI: l'unico stato era KV denso per OGNI layer (`kv::KvCache`,
//! architetture qwen2/llama). Le famiglie IBRIDE che servono a F5 (es.
//! Qwen3 DeltaNet) hanno uno stato MISTO: solo una frazione dei layer ha
//! attenzione con KV cache (es. 16 su 64), gli altri portano uno stato
//! ricorrente a dimensione fissa, indipendente dalla lunghezza della
//! sequenza.
//!
//! Perché ORA (roadmap v2, roadmap v2 del 25/08): rifattorizzare
//! stato+trait+runtime DOPO che server/hot-swap/generate toccano il modello
//! costa giorni; anticipare l'astrazione prima di chiudere F3b step 4 costa ore.
//!
//! Contratto (invariante invariato dai test F2.5):
//! `forward_cached(prefill) + forward_cached(decode)` == full-recompute.
//! Questa astrazione cambia CHI possiede i buffer, non la semantica del
//! forward: per le architetture dense tutto deve restare bit-identico.

/// Stato di UN singolo layer per la sequenza corrente.
pub enum LayerSeq {
    /// Attenzione: K e V pre-allocati `[max_seq × dim]`.
    /// Usato da ogni layer attention, full o parziale che sia.
    Kv(LayerKv),
    /// Stato ricorrente a dimensione fissa, indipendente da `max_seq`
    /// (es. matrice di transizione DeltaNet). Il layout esatto dei dati
    /// viene fissato quando F5 porta la prima architettura ricorrente;
    /// l'astrazione è: buffer piatto + shape, zero dipendenza da posizioni.
    Recurrent(RecurrentState),
}

/// Buffers K/V di un layer attention: `[max_seq × dim]` ciascuno.
pub struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    dim: usize,
}

/// Stato ricorrente: buffer piatto + shape logica.
pub struct RecurrentState {
    data: Vec<f32>,
    /// shape logica (es. `[n_heads, d_state, d_head]`). Solo documentale finché
    /// F5 non fissa il layout operativo.
    shape: [usize; 3],
}

/// Stato multi-layer di una sequenza in elaborazione.
///
/// Sostituisce `kv::KvCache` (che gestiva solo il caso tutti-KV).
/// La posizione assoluta della sequenza resta unica per tutti i layer:
/// in regime autoregressivo i layer avanzano insieme.
pub struct SeqState {
    layers: Vec<LayerSeq>,
    /// Numero di posizioni già scritte (lunghezza sequenza processata).
    pub len: usize,
}

impl LayerKv {
    pub fn new(dim: usize, max_seq: usize) -> Self {
        Self { k: vec![0.0; max_seq * dim], v: vec![0.0; max_seq * dim], dim }
    }

    fn start(&self, pos: usize) -> usize {
        pos * self.dim
    }

    /// Slice K dal punto di scrittura corrente (`pos * dim` in poi).
    pub(crate) fn k_from(&mut self, pos: usize) -> &mut [f32] {
        let start = self.start(pos);
        &mut self.k[start..]
    }
    /// Slice V dal punto di scrittura corrente.
    pub(crate) fn v_from(&mut self, pos: usize) -> &mut [f32] {
        let start = self.start(pos);
        &mut self.v[start..]
    }
    /// Slice K delle prime `n_positions` posizioni (zona lettura).
    pub(crate) fn k_upto(&self, n_positions: usize) -> &[f32] {
        &self.k[..n_positions * self.dim]
    }
    /// Slice V delle prime `n_positions` posizioni (zona lettura).
    pub(crate) fn v_upto(&self, n_positions: usize) -> &[f32] {
        &self.v[..n_positions * self.dim]
    }
}

impl RecurrentState {
    /// Stato azzerato con shape logica `[d0, d1, d2]`.
    pub fn zeroed(shape: [usize; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        Self { data: vec![0.0; n], shape }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Buffer piatto dello stato (lettura/scrittura diretta dal forward).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl SeqState {
    /// Stato generico da layer espliciti: il costrutto per le architetture
    /// MISTE (attenzione parziale + ricorrenza), punto d'arrivo di F5.
    pub fn from_layers(layers: Vec<LayerSeq>) -> Self {
        Self { layers, len: 0 }
    }

    /// Stato tutti-KV: il caso delle architetture dense (qwen2/llama).
    /// Identico alla vecchia `KvCache::new`; resta come scorciatoia perché
    /// è il caso dell'99% delle architetture attuali.
    pub fn all_kv(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
        Self::from_layers((0..n_layers).map(|_| LayerSeq::Kv(LayerKv::new(kv_dim, max_seq))).collect())
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Layer KV in sola lettura. Panica con messaggio pulito se il layer è
    /// ricorrente: è un errore di DISPATCH del modello, non di dati.
    pub(crate) fn layer_kv(&self, li: usize) -> &LayerKv {
        match &self.layers[li] {
            LayerSeq::Kv(kv) => kv,
            LayerSeq::Recurrent(_) => panic!(
                "seq_state: layer {li} è Recurrent ma il modello ha chiesto KV \
                 (dispatch architettura errato)"
            ),
        }
    }

    /// Layer KV mutabile (zona scrittura).
    pub(crate) fn layer_kv_mut(&mut self, li: usize) -> &mut LayerKv {
        match &mut self.layers[li] {
            LayerSeq::Kv(kv) => kv,
            LayerSeq::Recurrent(_) => panic!(
                "seq_state: layer {li} è Recurrent ma il modello ha chiesto KV \
                 (dispatch architettura errato)"
            ),
        }
    }

    /// Layer ricorrente mutabile. Speculare a `layer_kv_mut`.
    ///
    /// `allow(dead_code)`: nessun modello ricorrente esiste ancora (F5 porta
    /// il primo). È parte del contratto dell'astrazione MISTA e ha i suoi
    /// test; il warning sparirà al primo consumatore reale.
    #[allow(dead_code)]
    pub(crate) fn layer_recurrent_mut(&mut self, li: usize) -> &mut RecurrentState {
        match &mut self.layers[li] {
            LayerSeq::Recurrent(r) => r,
            LayerSeq::Kv(_) => panic!(
                "seq_state: layer {li} è Kv ma il modello ha chiesto Recurrent \
                 (dispatch architettura errato)"
            ),
        }
    }

    // ── API storica di kv::KvCache, stessa semantica al byte ──
    // qwen2::forward_cached le usa identiche a prima: lo scopo di questa
    // rifattorizzazione è cambiare il POSSESSORE dei buffer, non il codice
    // del forward verificato dalla parity.

    /// Slice K del layer dalla posizione di scrittura corrente in poi.
    pub(crate) fn layer_k_mut(&mut self, li: usize) -> &mut [f32] {
        let pos = self.len;
        self.layer_kv_mut(li).k_from(pos)
    }
    /// Slice V del layer dalla posizione di scrittura corrente in poi.
    pub(crate) fn layer_v_mut(&mut self, li: usize) -> &mut [f32] {
        let pos = self.len;
        self.layer_kv_mut(li).v_from(pos)
    }
    /// Slice K del layer per le prime `n_positions` posizioni.
    pub(crate) fn layer_k_upto(&self, li: usize, n_positions: usize) -> &[f32] {
        self.layer_kv(li).k_upto(n_positions)
    }
    /// Slice V del layer per le prime `n_positions` posizioni.
    pub(crate) fn layer_v_upto(&self, li: usize, n_positions: usize) -> &[f32] {
        self.layer_kv(li).v_upto(n_positions)
    }

    /// Avanza la lunghezza dopo aver scritto `n` nuove posizioni.
    pub(crate) fn advance(&mut self, n: usize) {
        self.len += n;
    }

    /// Azzera lo stato per una nuova sequenza sullo stesso bufferset.
    /// - layer KV: basta riportare `len` a 0 (la zona oltre `len` non è mai
    ///   letta, i dati stantii vengono sovrascritti);
    /// - layer ricorrenti: lo stato PORTA memoria tra token, va azzerato davvero.
    pub fn reset(&mut self) {
        self.len = 0;
        for l in &mut self.layers {
            if let LayerSeq::Recurrent(r) = l {
                r.data.fill(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kv_stessa_semantica_della_vecchia_kvcache() {
        // Replicando l'uso che fa qwen2: scrivi seq*kv_dim elementi, leggi
        // fino a pos+seq, avanza.
        let mut st = SeqState::all_kv(2, 4, 16);
        assert_eq!(st.n_layers(), 2);
        assert_eq!(st.len, 0);

        // prefill di 3 posizioni sul layer 0
        st.layer_k_mut(0)[..12].copy_from_slice(&[1.0; 12]);
        st.layer_v_mut(0)[..12].copy_from_slice(&[2.0; 12]);
        st.advance(3);

        assert_eq!(st.len, 3);
        // lettura fino a total=3: vede SOLO quanto scritto
        assert_eq!(st.layer_k_upto(0, 3), &[1.0; 12]);
        assert_eq!(st.layer_v_upto(0, 3), &[2.0; 12]);

        // decode di 1 posizione sul layer 1: la slice ritornata è GIA
        // offsettata alla posizione corrente (len=3) — si scrive da [..]
        let wk = st.layer_k_mut(1);
        wk[..4].copy_from_slice(&[9.0; 4]);
        // la zona precedente del layer 1 è intatta (zeroed)
        assert_eq!(&st.layer_k_upto(1, 3)[..12], &[0.0; 12]);
        assert_eq!(&st.layer_k_upto(1, 4)[12..], &[9.0; 4]);
    }

    #[test]
    fn stato_misto_kv_e_recurrent_costruito_da_layer_espliciti() {
        // Schema ibrido tipo Qwen3 DeltaNet: layer 0 = attenzione con KV,
        // layer 1 = ricorrente, shape [n_heads, d_state, d_head].
        let layers = vec![
            LayerSeq::Kv(LayerKv::new(8, 32)),
            LayerSeq::Recurrent(RecurrentState::zeroed([2, 16, 8])),
        ];
        let mut st = SeqState::from_layers(layers);
        assert_eq!(st.n_layers(), 2);

        // il layer KV risponde alle APIKV
        st.layer_k_mut(0)[..8].copy_from_slice(&[5.0; 8]);
        st.advance(1);
        assert_eq!(st.layer_k_upto(0, 1), &[5.0; 8]);

        // il layer ricorrente espone il suo buffer con shape intatta
        let shape = st.layer_recurrent_mut(1).shape();
        assert_eq!(shape, [2, 16, 8]);
        st.layer_recurrent_mut(1).data_mut()[0] = 42.0;

        // il reset azzera lo stato ricorrente MA la struttura resta
        st.reset();
        assert_eq!(st.len, 0);
        assert!(st.layer_recurrent_mut(1).data().iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "dispatch architettura errato")]
    fn accesso_kv_su_layer_ricorrente_panica_con_messaggio_chiaro() {
        let layers =
            vec![LayerSeq::Recurrent(RecurrentState::zeroed([1, 4, 4]))];
        let st = SeqState::from_layers(layers);
        let _ = st.layer_kv(0);
    }

    #[test]
    #[should_panic(expected = "dispatch architettura errato")]
    fn accesso_recurrent_su_layer_kv_panica_con_messaggio_chiaro() {
        let layers = vec![LayerSeq::Kv(LayerKv::new(4, 8))];
        let mut st = SeqState::from_layers(layers);
        let _ = st.layer_recurrent_mut(0);
    }
}
