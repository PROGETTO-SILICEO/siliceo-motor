//! Prefix caching lato server — AC-4 (docs/AGENTIC_CORE.md).
//!
//! I carichi agentici ripetono prefissi ENORMI: system prompt, tool definitions,
//! la storia della conversazione fino all'ultimo turno. Ricalcolare quel
//! prefill a ogni richiesta è lavoro bruciato. Il server tiene quindi una
//! piccola LRU di snapshot post-prefill (`SeqState` clonato subito dopo il
//! prefill): alla richiesta nuova trova lo snapshot con il prefisso comune più
//! lungo e lo passa come seme a `generate_tokens_seeded`, che ricalcola solo
//! il delta.
//!
//! Correttezza (invariante verificato dai test): per causalità, un prefill
//! delta sopra uno stato coerente produce gli STESSI logits del prefill pieno.
//! La coerenza dello stato è garantita qui: si sema solo snapshot il cui
//! token-vettore è PREFISSO ESATTO dei token richiesti, troncato alla
//! lunghezza dichiarata (`truncate`), con capacità sufficiente.
//!
//! Invalidazione: la cache vive dentro `Inner` → l'hot-swap del modello
//! (D6c) ne crea una nuova vuota. Nessuna voce può sopravvivere al modello
//! che l'ha prodotta.

use s_models::state::SeqState;

/// Sotto questa lunghezza non vale la pena ricordare: overhead puro.
const MIN_TOKENS_PER_MEMORIZZARE: usize = 8;

struct Voce {
    /// Token del prompt che ha prodotto lo snapshot (la CHIAVE è il prefisso).
    tokens: Vec<u32>,
    /// Stato post-prefill: copre ESATTAMENTE `tokens.len()` posizioni.
    stato: SeqState,
    usata: u64,
}

pub(crate) struct PrefixCache {
    voci: Vec<Voce>,
    max_voci: usize,
    tick: u64,
}

impl PrefixCache {
    pub fn new(max_voci: usize) -> Self {
        Self { voci: Vec::new(), max_voci, tick: 0 }
    }

    /// Lo snapshot col prefisso comune più lungo, pronto per il seme:
    /// - chiave = prefisso esatto dei token richiesti (confronto al byte);
    /// - troncato a `min(len, ids.len()-1)` come vuole il contratto di
    ///   `generate_tokens_seeded` (serve almeno un token da ricalcolare,
    ///   per avere i logits dell'ultimo);
    /// - capacità ESTESA al volo se lo snapshot è più corto della richiesta:
    ///   la conversazione dell'agente cresce, lo snapshot quasi sempre resta
    ///   corto (scoperto dal test live). Crescere preserva i dati scritti e
    ///   azzera solo la zona che il forward non legge mai.
    ///
    /// Stati ibridi (con layer ricorrenti, da F5/F6): riuso SOLO per estensione
    /// (n==len, nessun truncate). Il troncare a n<len è rifiutato perché lo
    /// stato ricorrente non è posizionale — per la query identica si rifa
    /// prefill pieno (un token in più costa meno di un secondo percorso).
    pub fn find(&mut self, ids: &[u32], capacita_richiesta: usize) -> Option<(SeqState, usize)> {
        let mut migliore: Option<usize> = None;
        for (i, v) in self.voci.iter().enumerate() {
            if v.tokens.len() > ids.len() || v.tokens[..] != ids[..v.tokens.len()] {
                continue;
            }
            if migliore.map_or(true, |m| self.voci[m].tokens.len() < v.tokens.len()) {
                migliore = Some(i);
            }
        }
        let i = migliore?;
        // l'uso conta per la LRU (e il tick avanza: due voci toccate nella
        // stessa chiamata a store() devono restare distinguibili)
        self.tick += 1;
        self.voci[i].usata = self.tick;
        let v = &self.voci[i];
        let n = v.tokens.len().min(ids.len().saturating_sub(1));
        if n == 0 {
            return None;
        }
        let mut stato = v.stato.clone();
        stato.truncate(n).ok()?;
        stato.ensure_capacity(capacita_richiesta);
        Some((stato, n))
    }

    /// Ricorda lo snapshot post-prefill di una richiesta (chiamata dalla
    /// callback `on_prefill` di `generate_tokens_seeded`).
    pub fn store(&mut self, tokens: Vec<u32>, stato: SeqState) {
        if tokens.len() < MIN_TOKENS_PER_MEMORIZZARE {
            return;
        }
        self.tick += 1;
        if let Some(v) = self.voci.iter_mut().find(|v| v.tokens == tokens) {
            v.stato = stato;
            v.usata = self.tick;
            return;
        }
        if self.voci.len() >= self.max_voci {
            let lru = self
                .voci
                .iter()
                .enumerate()
                .min_by_key(|(_, v)| v.usata)
                .map(|(i, _)| i);
            if let Some(i) = lru {
                self.voci.remove(i);
            }
        }
        self.voci.push(Voce { tokens, stato, usata: self.tick });
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.voci.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s_models::state::{LayerKv, LayerSeq};

    fn stato(dim: usize, max_seq: usize, len: usize) -> SeqState {
        let mut st = SeqState::all_kv(1, dim, max_seq);
        st.advance(len);
        st
    }

    #[test]
    fn trova_il_prefisso_piu_lungo_e_lo_tronca_al_contratto() {
        let mut c = PrefixCache::new(4);
        let corti: Vec<u32> = (0..10).collect();
        c.store(corti.clone(), stato(4, 64, 10));
        let lunghi: Vec<u32> = (0..20).collect();
        c.store(lunghi.clone(), stato(4, 64, 20));

        // richiesta che ESTENDE i lunghi: vince il prefisso più lungo (20).
        // Il taglio a len-1 serve SOLO quando la voce copre tutta la query:
        // qui restano token da ricalcolare (25-20) → n = 20 intero.
        let query: Vec<u32> = (0..25).collect();
        let (st, n) = c.find(&query, 30).unwrap();
        assert_eq!(n, 20);
        assert_eq!(st.len, 20, "lo stato consegnato è ALLINEATO alla n dichiarata");

        // richiesta IDENTICA alla voce memorizzata: n scende a len-1
        let identica: Vec<u32> = (0..20).collect();
        let (_, n2) = c.find(&identica, 30).unwrap();
        assert_eq!(n2, 19);
    }

    #[test]
    fn rifiuta_prefissi_non_esatti_e_capacita_insufficienti() {
        let mut c = PrefixCache::new(4);
        let chiave: Vec<u32> = (0..10).collect();
        c.store(chiave.clone(), stato(4, 16, 10));

        // prefisso NON esatto (differenza in mezzo) → nessun seme
        let mut sbagliata = chiave.clone();
        sbagliata[5] = 999;
        assert!(c.find(&sbagliata, 12).is_none());

        // capacità INSUFFICIENTE (snapshot 16 < richiesta 50): viene ESTESA
        // al volo preservando i dati — il riuso funziona comunque.
        // n = min(10, 15-1) = 10: il taglio a len-1 serve solo se la voce
        // copre l'intera query
        let estesa: Vec<u32> = (0..15).collect();
        let (st, n) = c.find(&estesa, 50).expect("estensione di capacità al volo");
        assert_eq!(n, 10);
        assert_eq!(st.len, 10);
        assert!(st.capacity() >= 50);
    }

    #[test]
    fn sotto_la_soglia_non_memorizza() {
        let mut c = PrefixCache::new(4);
        c.store(vec![1, 2, 3], stato(4, 16, 3));
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn lru_evoca_la_voce_meno_recentemente_usata() {
        let mut c = PrefixCache::new(2);
        let a: Vec<u32> = (0..8).collect();
        let b: Vec<u32> = (10..18).collect();
        let d: Vec<u32> = (20..28).collect();
        c.store(a.clone(), stato(4, 16, 8));
        c.store(b.clone(), stato(4, 16, 8));
        // tocchiamo `a`: diventa la più recente
        assert!(c.find(&a, 9).is_some());
        // terza voce → evita la meno recente, cioè `b`
        c.store(d.clone(), stato(4, 16, 8));
        assert_eq!(c.len(), 2);
        assert!(c.find(&b, 9).is_none(), "`b` doveva essere evocata");
        assert!(c.find(&a, 9).is_some());
        assert!(c.find(&d, 9).is_some());
    }

    #[test]
    fn stati_con_layer_ricorrenti_non_vengono_seminati() {
        let mut c = PrefixCache::new(4);
        let mut misto = SeqState::from_layers(vec![
            LayerSeq::Kv(LayerKv::new(4, 64)),
            LayerSeq::Recurrent(s_models::state::RecurrentState::zeroed([1, 4, 4])),
        ]);
        misto.advance(8);
        let chiave: Vec<u32> = (0..8).collect();
        c.store(chiave.clone(), misto);
        // F6: estensione del prefisso (n==len) OK anche per ibridi; truncamento (n<len) rifiutato
        let query_estesa: Vec<u32> = (0..12).collect();
        let (st, n) = c.find(&query_estesa, 14).expect("estensione ibrida deve riuscire (n==len)");
        assert_eq!(n, 8);
        assert_eq!(st.len, 8);
        // richiesta IDENTICA → n=len-1=7 → truncate rifiutato → nessun seme
        let query_identica: Vec<u32> = (0..8).collect();
        assert!(c.find(&query_identica, 14).is_none());
    }
}
