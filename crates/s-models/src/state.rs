//! Stato di sequenza generalizzato — F3b finestra 1.
//!
//! FINO A OGGI: l'unico stato era KV denso per OGNI layer (`kv::KvCache`,
//! architetture qwen2/llama). Le famiglie IBRIDE che servono a F5 (es.
//! Qwen3 DeltaNet) hanno uno stato MISTO: solo una frazione dei layer ha
//! attenzione con KV cache (es. 16 su 64), gli altri portano uno stato
//! ricorrente a dimensione fissa, indipendente dalla lunghezza della
//! sequenza.
//!
//! Perché ORA (roadmap v2 del 25/08): rifattorizzare
//! stato+trait+runtime DOPO che server/hot-swap/generate toccano il modello
//! costa giorni; anticipare l'astrazione prima di chiudere F3b step 4 costa ore.
//!
//! Contratto (invariante invariato dai test F2.5):
//! `forward_cached(prefill) + forward_cached(decode)` == full-recompute.
//! Questa astrazione cambia CHI possiede i buffer, non la semantica del
//! forward: per le architetture dense tutto deve restare bit-identico.

/// Stato di UN singolo layer per la sequenza corrente.
#[derive(Clone)]
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
#[derive(Clone)]
pub struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    dim: usize,
}

/// Stato ricorrente: buffer piatto + shape logica.
/// F6: per DeltaNet il buffer contiene DUE regioni contigue:
/// `[ S (d_state×d_state×n_v_heads) | conv_buf (conv_k×conv_dim) ]`
/// così Clone / reset / prefix-cache clonano TUTTO lo stato del layer in un colpo.
#[derive(Clone)]
pub struct RecurrentState {
    data: Vec<f32>,
    /// shape logica (es. `[n_heads, d_state, d_head]`). Solo documentale finché
    /// F5 non fissa il layout operativo.
    shape: [usize; 3],
    /// dimensione conv per DeltaNet (0 = stato puro senza convoluzione).
    conv_dim: usize,
    conv_k: usize,
}

/// Stato multi-layer di una sequenza in elaborazione.
///
/// Sostituisce `kv::KvCache` (che gestiva solo il caso tutti-KV).
/// La posizione assoluta della sequenza resta unica per tutti i layer:
/// in regime autoregressivo i layer avanzano insieme.
///
/// `Clone` serve al prefix caching del server (AC-4): lo snapshot
/// post-prefill è una copia completa dei buffer, riusabile come seme
/// per le richieste che condividono lo stesso prefisso.
#[derive(Clone)]
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
    /// Posizioni allocabili del layer (= il `max_seq` di costruzione).
    pub(crate) fn capacity(&self) -> usize {
        self.k.len() / self.dim
    }

    /// Cresce i buffer fino a ospitare `min_capacity` posizioni, PRESERVANDO
    /// i dati già scritti (le posizioni 0..len contengono K/V vere; oltre
    /// `len` non c'è mai niente da preservare perché il forward causale non
    /// legge mai la zona non scritta).
    ///
    /// Serve al riuso del prefix caching: uno snapshot dimensionato sulla
    /// richiesta precedente è quasi sempre troppo corto per la richiesta
    /// nuova (la conversazione dell'agente CRECE). Scoperto dal test live,
    /// non dagli unit test.
    pub(crate) fn ensure(&mut self, min_capacity: usize) {
        let cap = self.capacity();
        if min_capacity <= cap {
            return;
        }
        self.k.resize(min_capacity * self.dim, 0.0);
        self.v.resize(min_capacity * self.dim, 0.0);
    }
}

impl RecurrentState {
    /// Stato azzerato con shape logica `[d0, d1, d2]`.
    pub fn zeroed(shape: [usize; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        Self { data: vec![0.0; n], shape, conv_dim: 0, conv_k: 0 }
    }

    /// Stato DeltaNet: S [n_heads×d_state×d_state] + conv_buf [conv_k×conv_dim].
    /// shape = [n_v_heads, d_state, d_state] per S; conv_dim = key_dim*2+value_dim, conv_k=4.
    pub fn delta_net(shape: [usize; 3], conv_dim: usize, conv_k: usize) -> Self {
        let n_s = shape[0] * shape[1] * shape[2];
        let n_conv = conv_dim * conv_k;
        Self { data: vec![0.0; n_s + n_conv], shape, conv_dim, conv_k }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Buffer piatto dello stato (lettura/scrittura diretta dal forward).
    /// Per DeltaNet questo è SOLO la regione S (compatibile col codice F5).
    /// Usare `conv_buf_mut()` per la convoluzione.
    pub fn data_mut(&mut self) -> &mut [f32] {
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        &mut self.data[..n_s]
    }
    pub fn data(&self) -> &[f32] {
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        &self.data[..n_s]
    }

    /// Buffer piatto completo (S + conv). Usato da `reset()` e Clone.
    pub(crate) fn full_data(&self) -> &[f32] {
        &self.data
    }
    pub(crate) fn full_data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Conv buffer: regione dopo S, size `conv_k * conv_dim`.
    /// Ritorna slice vuota se `conv_dim==0` (stato puro non-DeltaNet).
    pub(crate) fn conv_buf(&self) -> &[f32] {
        if self.conv_dim == 0 {
            return &[];
        }
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        &self.data[n_s..]
    }
    pub(crate) fn conv_buf_mut(&mut self) -> &mut [f32] {
        if self.conv_dim == 0 {
            return &mut [];
        }
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        &mut self.data[n_s..]
    }

    /// Accesso diretto a S come mut (alias di data_mut, esplicito per F6).
    pub(crate) fn s_mut(&mut self) -> &mut [f32] {
        self.data_mut()
    }
    pub(crate) fn s(&self) -> &[f32] {
        self.data()
    }

    pub fn conv_dim(&self) -> usize { self.conv_dim }
    pub fn conv_k(&self) -> usize { self.conv_k }

    /// Entrambe le regioni mut in UN colpo (evita doppio borrow).
    pub(crate) fn parts_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        let (s, c) = self.data.split_at_mut(n_s);
        (s, c)
    }
    pub(crate) fn parts(&self) -> (&[f32], &[f32]) {
        let n_s = self.shape[0] * self.shape[1] * self.shape[2];
        let (s, c) = self.data.split_at(n_s);
        (s, c)
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
    ///
    /// Pubblico dal AC-4: il server (prefix cache) è un consumatore
    /// legittimo del contratto "chi possiede lo stato lo allinea" —
    /// il seme consegnato a generate_tokens_seeded deve avere
    /// `len() == n`, e allinearlo passa da qui.
    pub fn advance(&mut self, n: usize) {
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

    /// Posizioni massime ospitabili dallo stato: la più stretta tra i layer KV.
    /// Gli stati solo-ricorrenti non crescono con la sequenza → capacità infinita.
    #[allow(dead_code)] // consumato dal server (AC-4); il test lo copre
    pub fn capacity(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerSeq::Kv(kv) => kv.capacity(),
                LayerSeq::Recurrent(_) => usize::MAX,
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    /// Cresce lo stato fino a `min_capacity` posizioni (layer KV soltanto;
    /// quelli ricorrenti hanno dimensione fissa e non ne hanno bisogno).
    /// Vedi `LayerKv::ensure` per il perché e la garanzia sui dati.
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        for l in &mut self.layers {
            if let LayerSeq::Kv(kv) = l {
                kv.ensure(min_capacity);
            }
        }
    }

    /// Accorcia lo stato alle prime `len` posizioni processate.
    ///
    /// Valido SOLO su stati tutti-KV: le posizioni oltre `len` non vengono mai
    /// lette dal forward causale, quindi "dimenticare la coda" è gratis.
    /// Uno stato RICORRENTE porta memoria compatta non posizionale: NON può
    /// tornare a uno stato precedente — restituisce Err e non tocca nulla.
    ///
    /// È l'operazione che permette al prefix caching di riusare uno snapshot
    /// anche quando il prefisso comune è più corto dello snapshot stesso,
    /// purché lo stato sia troncabile.
    pub fn truncate(&mut self, len: usize) -> Result<(), String> {
        if len > self.len {
            return Err(format!(
                "truncate: {len} > len={} (allungare uno stato è senza senso)",
                self.len
            ));
        }
        if len == self.len {
            return Ok(());
        }
        if self.layers.iter().any(|l| matches!(l, LayerSeq::Recurrent(_))) {
            return Err(
                "truncate: lo stato ha layer ricorrenti, che non possono tornare \
                 indietro (la loro memoria non è posizionale)"
                    .to_string(),
            );
        }
        self.len = len;
        Ok(())
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

    // ── AC-4 prefix caching: clone, capacity, truncate ──

    #[test]
    fn capacity_e_il_max_seq_dei_layer_kv() {
        assert_eq!(SeqState::all_kv(3, 4, 16).capacity(), 16);
        // solo ricorrente: capacità infinita (lo stato non cresce con la seq)
        let rec = vec![LayerSeq::Recurrent(RecurrentState::zeroed([1, 8, 8]))];
        assert_eq!(SeqState::from_layers(rec).capacity(), usize::MAX);
        // misto: la più stretta tra i layer KV
        let misto = vec![
            LayerSeq::Kv(LayerKv::new(4, 32)),
            LayerSeq::Recurrent(RecurrentState::zeroed([1, 8, 8])),
            LayerSeq::Kv(LayerKv::new(4, 8)),
        ];
        assert_eq!(SeqState::from_layers(misto).capacity(), 8);
    }

    #[test]
    fn truncate_accorcia_lo_stato_tutti_kv() {
        let mut st = SeqState::all_kv(2, 4, 16);
        st.layer_k_mut(0)[..12].copy_from_slice(&[1.0; 12]);
        st.advance(3);
        st.truncate(1).unwrap();
        assert_eq!(st.len, 1);
        // la zona oltre len non è letta dal forward causale: qui verifichiamo
        // solo che il contratto posizionale regga — riscrivere da len=1 funziona
        st.layer_k_mut(0)[..4].copy_from_slice(&[7.0; 4]);
        assert_eq!(st.layer_k_upto(0, 2)[4..8], [7.0; 4]);
    }

    #[test]
    fn truncate_rifiuta_stati_con_layer_ricorrenti_senza_toccarli() {
        let mut st = SeqState::from_layers(vec![
            LayerSeq::Kv(LayerKv::new(4, 8)),
            LayerSeq::Recurrent(RecurrentState::zeroed([1, 4, 4])),
        ]);
        st.advance(5);
        let err = st.truncate(2).unwrap_err();
        assert!(err.contains("ricorrenti"), "messaggio fuorviante: {err}");
        // lo stato resta INTATTO dopo il rifiuto
        assert_eq!(st.len, 5);
    }

    #[test]
    fn truncate_ricorrente_identita_ok_per_prefix_cache_f6() {
        let mut st = SeqState::from_layers(vec![
            LayerSeq::Kv(LayerKv::new(4, 8)),
            LayerSeq::Recurrent(RecurrentState::zeroed([1, 4, 4])),
        ]);
        st.advance(5);
        // n==len è no-op, deve riuscire anche per ibridi (permetta riuso esatto)
        st.truncate(5).unwrap();
        assert_eq!(st.len, 5);
    }

    #[test]
    fn truncate_rifiuta_allungamenti() {
        let mut st = SeqState::all_kv(1, 4, 8);
        st.advance(2);
        assert!(st.truncate(3).is_err());
        assert_eq!(st.len, 2);
    }

    #[test]
    fn clone_indipendente_prefix_cache() {
        let mut st = SeqState::all_kv(1, 4, 16);
        st.layer_k_mut(0)[..8].copy_from_slice(&[1.0; 8]);
        st.advance(2);
        let snap = st.clone();
        // l'originale avanza: lo snapshot NON deve muoversi con lui.
        // (la slice di layer_k_mut è GIÀ offsettata a len*dim → [..4] = pos 2)
        st.layer_k_mut(0)[..4].copy_from_slice(&[9.0; 4]);
        st.advance(1);
        assert_eq!(snap.len, 2);
        assert_eq!(&snap.layer_k_upto(0, 2)[..8], &[1.0; 8]);
        assert_eq!(st.len, 3);
        assert_eq!(st.layer_k_upto(0, 3)[8..12], [9.0; 4]);
    }
}
