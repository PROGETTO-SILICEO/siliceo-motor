//! Streaming SSE — AC-1 (docs/AGENTIC_CORE.md).
//!
//! Trasporto: `Content-Type: text/event-stream` + `Connection: close`.
//! La risposta è delimitata da EOF (niente chunked encoding da gestire a
//! mano): compatibile con curl e con i client OpenAI.
//!
//! Contenuto: un evento per PEZZO DI TESTO NUOVO. Il testo non viene mai
//! spezzato a metà carattere UTF-8 perché si calcola come diff del decode
//! dell'intera sequenza generata (sempre una stringa valida).
//!
//! Stop strings: mentre la richiesta dichiara stop, si TRATTIENE una coda
//! lunga quanto lo stop più lungo (meno uno): se il testo finisce dentro
//! una stop string, al client non è già arrivato troppo. Senza stop, tutto
//! viene inviato appena nasce.

/// Calcola i pezzi sicuri da inviare dato il testo completo decodificato.
pub struct StreamSlicer {
    /// numero di CHAR già inviati (char = scalare Unicode: sempre boundary-safe)
    sent_chars: usize,
    /// char da trattenere in coda (= max lunghezza stop); 0 senza stop.
    /// Con coda di L char, quando lo stop di L char compare il suo PRIMO
    /// carattere non è ancora stato emesso (safe_end = inizio dello stop):
    /// zero leak, al costo di un char di latenza.
    hold_chars: usize,
}

impl StreamSlicer {
    pub fn new(stop: &[String]) -> Self {
        let longest = stop.iter().map(|s| s.chars().count()).max().unwrap_or(0);
        Self { sent_chars: 0, hold_chars: longest }
    }

    /// Nuovo pezzo inviabile dato il testo completo corrente. Può essere "".
    pub fn push(&mut self, full: &str) -> String {
        let total = full.chars().count();
        if total <= self.sent_chars {
            return String::new();
        }
        let safe_end = total.saturating_sub(self.hold_chars);
        if safe_end <= self.sent_chars {
            return String::new();
        }
        let piece: String =
            full.chars().skip(self.sent_chars).take(safe_end - self.sent_chars).collect();
        self.sent_chars = safe_end;
        piece
    }

    /// Coda residua a fine generazione (già troncata alle stop dal chiamante).
    pub fn finish(&mut self, full: &str) -> String {
        let total = full.chars().count();
        if total <= self.sent_chars {
            return String::new();
        }
        let piece: String = full.chars().skip(self.sent_chars).collect();
        self.sent_chars = total;
        piece
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn senza_stop_tutto_viene_inviato_appena_nasce() {
        let mut sl = StreamSlicer::new(&[]);
        assert_eq!(sl.push("La cap"), "La cap");
        assert_eq!(sl.push("La capitale"), "itale");
        assert_eq!(sl.finish("La capitale."), ".");
    }

    #[test]
    fn utf8_spezzato_mai_emesso_a_meta_carattere() {
        // Il decode dell'intera sequenza produce SEMPRE stringhe valide:
        // quando un char multibyte è completo, il diff lo emette intero.
        let mut sl = StreamSlicer::new(&[]);
        assert_eq!(sl.push("pi"), "pi");
        assert_eq!(sl.push("pi\u{00E8}"), "\u{00E8}"); // 'è' arriva intero
    }

    #[test]
    fn con_stop_la_coda_e_trattenuta() {
        let mut sl = StreamSlicer::new(&["FINE".to_string()]);
        // coda trattenuta = 4 char (len): nessun char di un potenziale stop
        // viene emesso prima che lo stop sia rilevato dal loop
        assert_eq!(sl.push("abc"), "");          // safe_end = 0
        assert_eq!(sl.push("abc FI"), "ab");     // safe_end = 6-4 = 2
        assert_eq!(sl.push("abc FIN"), "c");     // safe_end = 7-4 = 3, nuovo = 1
        // NB: finish riceve il testo GIÀ TRONCATO alle stop dal chiamante
        // (Generated.text) — qui simuliamo il flusso grezzo
        assert_eq!(sl.finish("abc FINALE!"), " FINALE!");
    }

    #[test]
    fn lo_zero_leak_dello_stop_e_garantito() {
        // il testo cresce fino a CONTENERE lo stop: il primo char dello stop
        // non deve mai essere stato emesso prima del rilevamento
        let mut sl = StreamSlicer::new(&["capitale".to_string()]);
        let mut inviato = String::new();
        for full in ["La", "La capit", "La capitale"] {
            inviato.push_str(&sl.push(full));
        }
        assert!(!inviato.contains('c') || !inviato.ends_with("La c"),
            "emesso parte dello stop: {inviato:?}");
        assert_eq!(inviato, "La ");
    }

    #[test]
    fn testo_identico_non_riemette_nulla() {
        let mut sl = StreamSlicer::new(&[]);
        assert_eq!(sl.push("x"), "x");
        assert_eq!(sl.push("x"), "");
        assert_eq!(sl.finish("x"), "");
    }
}
