//! JSON minimale per s-server — parser e serializzatore scritti in casa.
//!
//! Scope F2: basta per le richieste OpenAI-compatible
//! (`messages`, `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`)
//! e per comporre le risposte. Niente numeri esponenziali in parsing
//! (non servono per i parametri previsti), escape \u completo con surrogate.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|n| n as usize)
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }

    /// Serializza in JSON compatto. (Usato dai test; il server compone
    /// le risposte a mano per controllo fine dell'output.)
    #[allow(dead_code)]
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    #[allow(dead_code)]
    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 9e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Json::Str(s) => write_escaped(out, s),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Escape di una stringa JSON (\", \\, controllo, \uXXXX con surrogate pair).
pub fn write_escaped(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ── parser ──

pub fn parse(input: &str) -> Result<Json, String> {
    let bytes = input.as_bytes();
    let mut p = Parser { b: bytes, i: 0 };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.i != bytes.len() {
        return Err(format!("contenuto inatteso dopo il JSON alla posizione {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek().ok_or("JSON troncato")? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => {
                if self.eat("true") {
                    Ok(Json::Bool(true))
                } else {
                    Err("valore non valido (atteso true)".into())
                }
            }
            b'f' => {
                if self.eat("false") {
                    Ok(Json::Bool(false))
                } else {
                    Err("valore non valido (atteso false)".into())
                }
            }
            b'n' => {
                if self.eat("null") {
                    Ok(Json::Null)
                } else {
                    Err("valore non valido (atteso null)".into())
                }
            }
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("attesa chiave stringa alla posizione {}", self.i));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("atteso ':' dopo la chiave alla posizione {}", self.i));
            }
            self.i += 1;
            self.skip_ws();
            let val = self.value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err("atteso ',' o '}' nell'oggetto".into()),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err("atteso ',' o ']' nell'array".into()),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // '"'
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("stringa non chiusa")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or("escape troncato")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0C'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // surrogate pair
                                if !self.eat("\\u") {
                                    return Err("surrogate basso mancante".into());
                                }
                                let lo = self.hex4()?;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or("codepoint non valido")?);
                        }
                        _ => return Err("escape non riconosciuto".into()),
                    }
                }
                _ => {
                    // consumiamo l'intero carattere UTF-8 multibyte se serve
                    let start = self.i - 1;
                    let len = utf8_len(c);
                    self.i = start + len;
                    let s = std::str::from_utf8(&self.b[start..self.i])
                        .map_err(|_| "utf8 non valido nel body")?;
                    out.push_str(s);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("\\u troncato".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|_| "\\u non valido".to_string())?;
        let v = u32::from_str_radix(s, 16)
            .map_err(|_| "\\u non esadecimale".to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while self
            .peek()
            .map(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
            .unwrap_or(false)
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "numero non valido".to_string())?;
        s.parse::<f64>().map(Json::Num).map_err(|_| format!("numero non valido: {s:?}"))
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chat_request() {
        let src = r#"{"model":"qwen","messages":[
            {"role":"system","content":"Sei utile."},
            {"role":"user","content":"Ciao \"mondo\"\nè tardi"}
        ],"temperature":0.7,"top_p":0.9,"max_tokens":128,"stop":["FINE"]}"#;
        let j = parse(src).unwrap();
        assert_eq!(j.get("model").unwrap().as_str(), Some("qwen"));
        let msgs = j.get("messages").unwrap().as_arr().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].get("role").unwrap().as_str(), Some("system"));
        assert_eq!(
            msgs[1].get("content").unwrap().as_str(),
            Some("Ciao \"mondo\"\nè tardi")
        );
        assert!((j.get("temperature").unwrap().as_f64().unwrap() - 0.7).abs() < 1e-9);
        assert_eq!(j.get("max_tokens").unwrap().as_usize(), Some(128));
        assert_eq!(j.get("top_k"), None); // assente → None
    }

    #[test]
    fn roundtrip_unicode_e_escape() {
        let orig = "accenti àèìòù emoji \u{1F600} newline\n tab\t";
        let mut buf = String::new();
        write_escaped(&mut buf, orig);
        let parsed = parse(&buf).unwrap();
        assert_eq!(parsed.as_str(), Some(orig));
    }

    #[test]
    fn errori_puliti() {
        assert!(parse("{").is_err());
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("[1,2").is_err());
        assert!(parse("{} extra").is_err());
    }

    #[test]
    fn dump_num_intero_senza_decimale() {
        assert_eq!(Json::Num(42.0).dump(), "42");
        assert_eq!(Json::Num(0.5).dump(), "0.5");
    }
}
