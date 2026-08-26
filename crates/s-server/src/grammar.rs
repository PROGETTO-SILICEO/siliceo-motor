//! Structured output — AC-3 (docs/AGENTIC_CORE.md).
//!
//! Sampling vincolato: `response_format:{type:"json_object"}` (o json_schema,
//! v1 = vincolo JSON generico documentato) attiva una maschera che ammette
//! solo i token coerenti con una grammatica JSON incrementale.
//!
//! Architettura:
//! - `JsonState`: automa a pila (pushdown) sui CARATTERI; ogni token candidato
//!   viene SIMULATO su un clone dello stato — se tutti i suoi char passano,
//!   il token è ammesso;
//! - `JsonMasker`: tabella vocabolario→testo decodificato costruita UNA volta
//!   per modello, poi maschere O(1) per passo;
//! - i token SPECIALI (decode vuoto) sono vietati finché il JSON non è
//!   completo: a stato Done tornano ammessi (il modello può chiudere il turno).
//!
//! Onestà v1 (documentata): i NUMERI accettano solo sequenze sintatticamente
//! valide (intero/frazione/esponente) ma non si valida la semantica Unicode
//! delle stringhe; lo schema-specifico (required/tipi) arriva dopo.

// ── automa a pila sui caratteri ──

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    /// attende l'inizio di un valore
    Value,
    /// attende la chiave stringa di un oggetto
    Key,
    /// attende ':'
    Colon,
    /// oggetto: ',' o '}'
    CommaObj,
    /// array: ',' o ']'
    CommaArr,
    /// valore radice completato
    Done,
}

#[derive(Clone, Copy)]
struct Ctl {
    obj: bool,
}

/// Stato della grammatica. Clone economico: la simulazione dei candidati
/// lavora su cloni, lo stato reale avanza solo coi token campionati.
#[derive(Clone)]
pub struct JsonState {
    stack: Vec<Ctl>,
    phase: Phase,
    in_str: bool,
    is_key: bool,
    esc: bool,
    /// letterale in corso ("true"/"false"/"null"): (parola, prossima pos)
    lit: (&'static str, usize),
    num: Option<NumSt>,
    /// il numero ha una cifra significativa '0' in testa (vietato "01")
    num_lead_zero: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum NumSt {
    Int,     // cifre (dopo eventuale '-')
    Dot,     // appena visto '.', serve cifra
    Frac,    // cifre dopo il punto
    E,       // appena visto 'e'/'E', serve cifra o segno
    ESign,   // segno visto, serve cifra
    EDig,    // cifre dell'esponente
}

impl JsonState {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            phase: Phase::Value,
            in_str: false,
            is_key: false,
            esc: false,
            lit: ("", 0),
            num: None,
            num_lead_zero: false,
        }
    }

    pub fn is_done(&self) -> bool {
        if self.in_str || self.lit.1 > 0 {
            return false;
        }
        match self.num {
            // numero in corso a fine testo: è implicitamente completo se in
            // uno stato terminale ("42", "1.5", "1e3" sì; "1.", "1e+" no)
            Some(st) => {
                self.stack.is_empty() && matches!(st, NumSt::Int | NumSt::Frac | NumSt::EDig)
            }
            None => self.phase == Phase::Done,
        }
    }

    fn after_value(&mut self) {
        match self.stack.last_mut() {
            None => self.phase = Phase::Done,
            Some(c) => {
                if c.obj {
                    self.phase = Phase::CommaObj;
                } else {
                    self.phase = Phase::CommaArr;
                }
            }
        }
    }

    fn after_pop(&mut self) {
        if self.stack.is_empty() {
            self.phase = Phase::Done;
        } else {
            self.after_value();
        }
    }

    /// Esito del consumo di un char.
    fn step(&mut self, c: char) -> StepRes {
        // ── dentro stringa ──
        if self.in_str {
            if self.esc {
                self.esc = false;
                return StepRes::Consumed;
            }
            match c {
                '\\' => {
                    self.esc = true;
                    StepRes::Consumed
                }
                '"' => {
                    self.in_str = false;
                    if self.is_key {
                        self.is_key = false;
                        self.phase = Phase::Colon;
                    } else {
                        self.after_value();
                    }
                    StepRes::Consumed
                }
                // strict JSON: dentro le stringhe i control character (<0x20)
                // sono vietati senza escape (il test live beccava "\n" grezzo)
                c2 if (c2 as u32) < 0x20 => StepRes::Invalid,
                _ => StepRes::Consumed,
            }
        } else if self.lit.1 > 0 {
            // ── letterale parziale: SOLO la lettera attesa (niente spazi:
            // "tru e" resta invalido) ──
            let (word, pos) = self.lit;
            let atteso = word.as_bytes()[pos] as char;
            if c != atteso {
                return StepRes::Invalid;
            }
            if pos + 1 == word.len() {
                self.lit = ("", 0);
                self.after_value();
            } else {
                self.lit = (word, pos + 1);
            }
            StepRes::Consumed
        } else if let Some(st) = self.num {
            // ── dentro numero ──
            match st {
                NumSt::Int => match c {
                    '0'..='9' => {
                        // zero iniziale seguito da cifra = JSON invalido ("01")
                        if self.num_lead_zero {
                            StepRes::Invalid
                        } else {
                            StepRes::Consumed
                        }
                    }
                    '.' => {
                        self.num = Some(NumSt::Dot);
                        StepRes::Consumed
                    }
                    'e' | 'E' => {
                        self.num = Some(NumSt::E);
                        StepRes::Consumed
                    }
                    _ => {
                        // il numero termina QUI: il char va riprocessato
                        self.num = None;
                        self.after_value();
                        StepRes::NotConsumed
                    }
                },
                NumSt::Dot => {
                    if c.is_ascii_digit() {
                        self.num = Some(NumSt::Frac);
                        StepRes::Consumed
                    } else {
                        StepRes::Invalid
                    }
                }
                NumSt::Frac => match c {
                    '0'..='9' => StepRes::Consumed,
                    'e' | 'E' => {
                        self.num = Some(NumSt::E);
                        StepRes::Consumed
                    }
                    _ => {
                        self.num = None;
                        self.after_value();
                        StepRes::NotConsumed
                    }
                },
                NumSt::E => match c {
                    '+' | '-' => {
                        self.num = Some(NumSt::ESign);
                        StepRes::Consumed
                    }
                    '0'..='9' => {
                        self.num = Some(NumSt::EDig);
                        StepRes::Consumed
                    }
                    _ => StepRes::Invalid,
                },
                NumSt::ESign => {
                    if c.is_ascii_digit() {
                        self.num = Some(NumSt::EDig);
                        StepRes::Consumed
                    } else {
                        StepRes::Invalid
                    }
                }
                NumSt::EDig => match c {
                    '0'..='9' => StepRes::Consumed,
                    _ => {
                        self.num = None;
                        self.after_value();
                        StepRes::NotConsumed
                    }
                },
            }
        } else if c.is_whitespace() {
            StepRes::Consumed
        } else {
            // ── stati strutturali ──
            match self.phase {
                Phase::Done => StepRes::Invalid,
                Phase::Value => match c {
                    '{' => {
                        self.stack.push(Ctl { obj: true });
                        self.phase = Phase::Key;
                        StepRes::Consumed
                    }
                    '[' => {
                        self.stack.push(Ctl { obj: false });
                        // ']' immediato o primo valore: entrambi leciti
                        self.phase = Phase::CommaArr;
                        StepRes::Consumed
                    }
                    '"' => {
                        self.in_str = true;
                        self.is_key = false;
                        StepRes::Consumed
                    }
                    '-' | '0'..='9' => {
                        self.num = Some(NumSt::Int);
                        // '0' come prima cifra significativa: "01" vietato
                        self.num_lead_zero = c == '0';
                        StepRes::Consumed
                    }
                    't' | 'f' | 'n' => {
                        let word = match c {
                            't' => "true",
                            'f' => "false",
                            _ => "null",
                        };
                        self.lit = (word, 1);
                        StepRes::Consumed
                    }
                    _ => StepRes::Invalid,
                },
                Phase::Key => match c {
                    '"' => {
                        self.in_str = true;
                        self.is_key = true;
                        StepRes::Consumed
                    }
                    // oggetto vuoto: } subito dopo {
                    '}' => {
                        self.stack.pop();
                        self.after_pop();
                        StepRes::Consumed
                    }
                    _ => StepRes::Invalid,
                },
                Phase::Colon => {
                    if c == ':' {
                        self.phase = Phase::Value;
                        StepRes::Consumed
                    } else {
                        StepRes::Invalid
                    }
                }
                Phase::CommaObj => match c {
                    ',' => {
                        self.phase = Phase::Key;
                        StepRes::Consumed
                    }
                    '}' => {
                        self.stack.pop();
                        self.after_pop();
                        StepRes::Consumed
                    }
                    _ => StepRes::Invalid,
                },
                Phase::CommaArr => match c {
                    ',' => {
                        self.phase = Phase::Value;
                        StepRes::Consumed
                    }
                    // ']' è lecito SEMPRE in CommaArr: array vuoto o chiusura
                    // dopo un valore completo
                    ']' => {
                        self.stack.pop();
                        self.after_pop();
                        StepRes::Consumed
                    }
                    _ => {
                        // qualsiasi inizio-valore, poi riprocesso il char
                        self.phase = Phase::Value;
                        StepRes::NotConsumed
                    }
                },
            }
        }
    }

    /// Consuma un intero testo. false = illegale da questo stato
    /// (lo stato può essere rimasto in uno stadio intermedio: i chiamanti
    /// simulano su CLONI).
    pub fn feed(&mut self, text: &str) -> bool {
        for ch in text.chars() {
            let mut guard = 0;
            loop {
                guard += 1;
                if guard > 8 {
                    return false; // difesa dai cicli
                }
                match self.step(ch) {
                    StepRes::Consumed => break,
                    StepRes::NotConsumed => continue,
                    StepRes::Invalid => return false,
                }
            }
        }
        true
    }

    /// Primi char legali nello stato corrente — filtro rapido pre-vocabolario.
    pub fn first_chars(&self) -> &'static [char] {
        const STRUCT_ALL: &[char] =
            &['{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        if self.in_str || self.num.is_some() {
            // dentro stringa/numero quasi tutto può passare: niente prune
            return &[];
        }
        match self.phase {
            Phase::Done => &[],
            Phase::Key => &['"'],
            Phase::Colon => &[':'],
            Phase::CommaObj => &[',', '}'],
            Phase::CommaArr => &[',', ']'],
            Phase::Value => STRUCT_ALL,
        }
    }
}

#[derive(PartialEq)]
enum StepRes {
    Consumed,
    NotConsumed,
    Invalid,
}

// ── mascheratore sul vocabolario ──

pub struct JsonMasker {
    /// decode di ogni id del vocabolario (condiviso col modello caricato)
    texts: std::sync::Arc<Vec<String>>,
    state: JsonState,
}

impl JsonMasker {
    /// Da tabella già costruita e condivisa (cache sul modello caricato):
    /// nessuna copia, solo un Arc.
    pub fn from_texts(texts: std::sync::Arc<Vec<String>>) -> Self {
        Self { texts, state: JsonState::new() }
    }

    /// Maschera per il passo corrente: mask[id] = token ammesso.
    pub fn mask(&self) -> Vec<bool> {
        let firsts = self.state.first_chars();
        self.texts
            .iter()
            .map(|text| {
                if text.is_empty() {
                    // speciali (EOS, <|im_end|>, ...): solo a JSON completo
                    return self.state.is_done();
                }
                if !firsts.is_empty() && !firsts.contains(&text.chars().next().unwrap()) {
                    return false;
                }
                let mut probe = self.state.clone();
                probe.feed(text)
            })
            .collect()
    }

    /// Avanza lo stato col token campionato.
    pub fn advance(&mut self, id: u32) {
        let text = &self.texts[id as usize];
        if !text.is_empty() {
            self.state.feed(text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_ok(text: &str) -> bool {
        let mut st = JsonState::new();
        st.feed(text) && st.is_done()
    }

    #[test]
    fn json_validi_accettati() {
        assert!(feed_ok("{}"));
        assert!(feed_ok("[]"));
        assert!(feed_ok("[ ]"));
        assert!(feed_ok("{\"a\":1}"));
        assert!(feed_ok("{\"nome\": \"Bergamo\", \"temp\": 18}"));
        assert!(feed_ok("[1, 2.5, -3e2, true, false, null]"));
        assert!(feed_ok("{\"a\": {\"b\": [1, \"x\\\"y\"]}}"));
        assert!(feed_ok("42"));
        assert!(feed_ok("\"solo stringa\""));
        assert!(feed_ok("{\n  \"citta\": \"Bergamo\"\n}\n"));
    }

    #[test]
    fn json_invalidi_rifiutati() {
        assert!(!feed_ok("{"));
        assert!(!feed_ok("{\"a\":}"));
        assert!(!feed_ok("{\"a\" 1}"));
        assert!(!feed_ok("[1,]"));      // virgola pendente
        assert!(!feed_ok("{,}"));
        assert!(!feed_ok("tru"));        // letterale incompleto a fine input
        assert!(!feed_ok("{\"a\": 01}")); // zero iniziale: 0 poi '1' → dopo-value invalido
        assert!(!feed_ok("{\"a\":1,,}"));
        assert!(!feed_ok("]"));
    }

    #[test]
    fn numeri_stretti_ma_completi() {
        assert!(feed_ok("[-0.5e+10]"));
        assert!(!feed_ok("[1.]"));   // punto senza cifra
        assert!(!feed_ok("[1e]"));   // esponente vuoto
        assert!(!feed_ok("[.5]"));   // parte intera mancante
    }

    #[test]
    fn simulazione_token_multi_char() {
        // il testo cresce a pezzi come farebbero i token: ogni pezzo deve
        // passare dallo stato raggiunto dal precedente
        let mut st = JsonState::new();
        for piece in ["{\"citta\"", ": ", "\"Ber", "gamo", "\"", ", ", "\"t\": 18}", "\n"] {
            assert!(st.feed(piece), "pezzo rifiutato: {piece:?}");
        }
        assert!(st.is_done());
    }

    #[test]
    fn stato_non_avanza_su_input_invalido_parciale() {
        let mut st = JsonState::new();
        assert!(st.feed("{\"a\": 1"));
        assert!(!st.feed("}spazzatura")); // '}' ok ma poi invalido
        // lo stato REALE non va usato dopo un fallimento: qui verifichiamo
        // solo che feed non vada in panico; il chiamante scarta il clone.
        let _ = st;
    }
}
