//! Config a strati — D6.
//!
//! Due famiglie di tipi:
//! - `FileConfig` / sezioni: campi Option, rappresentano "ciò che uno strato
//!   vuole cambiare". Il merge è `Option::or` in cascata.
//! - `GenerateDefaults` risolto: valori concreti usati dal server come punto
//!   di partenza per ogni richiesta (la richiesta può ancora override).

use crate::json::{parse, Json};

// ── default di generazione ──

/// Default di generazione. Serve in due vesti:
/// - **risolto** (campi pieni): i default correnti del server;
/// - **parziale** (campi None): uno strato che cambia solo qualcosa
///   (file config, patch runtime). `merge` li compone.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateDefaults {
    /// Temperatura (0 = greedy).
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
}

impl Default for GenerateDefaults {
    fn default() -> Self {
        // I default ASSOLUTI del motore (strato 1): greedy, deterministico,
        // come da F2. Gli altri strati possono cambiarli.
        Self { temperature: Some(0.0), top_p: None, top_k: None, max_tokens: Some(256) }
    }
}

impl GenerateDefaults {
    /// Fusione strati: `self` è più debole, `stronger` vince dove è specificato.
    pub fn merge(&self, stronger: &Self) -> Self {
        Self {
            temperature: stronger.temperature.or(self.temperature),
            top_p: stronger.top_p.or(self.top_p),
            top_k: stronger.top_k.or(self.top_k),
            max_tokens: stronger.max_tokens.or(self.max_tokens),
        }
    }

    fn from_json(j: &Json) -> Result<Self, String> {
        let mut g = Self { temperature: None, top_p: None, top_k: None, max_tokens: None };
        if let Some(v) = j.get("temperature") {
            let t = v.as_f64().ok_or("generate.temperature non numerica")? as f32;
            if !(0.0..=2.0).contains(&t) {
                return Err(format!("generate.temperature fuori range [0,2]: {t}"));
            }
            g.temperature = Some(t);
        }
        // esplicitare null = "non toccare" (utile nelle patch)
        if let Some(v) = j.get("top_p").filter(|v| !matches!(v, Json::Null)) {
            g.top_p = Some(v.as_f64().ok_or("generate.top_p non numerico")? as f32);
        }
        if let Some(v) = j.get("top_k").filter(|v| !matches!(v, Json::Null)) {
            g.top_k = Some(v.as_usize().ok_or("generate.top_k non intero")?);
        }
        if let Some(v) = j.get("max_tokens") {
            g.max_tokens = Some(v.as_usize().ok_or("generate.max_tokens non intero")?);
        }
        Ok(g)
    }

    /// Patch runtime da JSON parziale (POST /v1/config).
    pub fn patch_from_str(src: &str) -> Result<Self, String> {
        let j = parse(src).map_err(|e| format!("JSON non valido: {e}"))?;
        Self::from_json(&j)
    }
}

// ── sezioni del file ──

/// Sezione `server` del file config.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerDefaults {
    pub host: Option<String>,
    pub port: Option<u16>,
}

/// Config letta da UN file (o da un oggetto JSON qualsiasi).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileConfig {
    pub server: ServerDefaults,
    pub generate: GenerateDefaults,
    /// Path del modello; sovrascrive quello posizionale se presente.
    pub model_path: Option<String>,
    /// Template di chat: "auto" | "chatml" | "llama3". None = auto.
    pub model_template: Option<String>,
}

impl FileConfig {
    /// Parse da testo JSON. Accetta l'intera struttura:
    /// `{"server":{...},"model":{"path":"..."},"generate":{...}}`
    /// ma anche solo `{"temperature": 0.7}` (scorciatoia per la sola generazione).
    pub fn parse(src: &str) -> Result<Self, String> {
        let j = parse(src).map_err(|e| format!("config non valida: {e}"))?;
        Json::to_config(&j)
    }

    pub fn from_json(j: &Json) -> Result<Self, String> {
        Json::to_config(j)
    }
}

impl Json {
    fn to_config(j: &Json) -> Result<FileConfig, String> {
        if j.as_obj().is_none() {
            return Err("la config deve essere un oggetto JSON".into());
        }
        let mut c = FileConfig::default();

        if let Some(s) = j.get("server") {
            c.server.host = s.get("host").and_then(|v| v.as_str()).map(String::from);
            if let Some(v) = s.get("port") {
                c.server.port = Some(
                    v.as_usize()
                        .filter(|p| *p > 0 && *p <= 65535)
                        .ok_or("server.port deve essere 1..65535")? as u16,
                );
            }
        }

        if let Some(m) = j.get("model") {
            c.model_path = m.get("path").and_then(|v| v.as_str()).map(String::from);
            c.model_template =
                m.get("template").and_then(|v| v.as_str()).map(String::from);
            // chiavi non riconosciute nella sezione model sono tollerate
            // (context arriva in D6c) — nessun errore.
        }

        if let Some(g) = j.get("generate") {
            c.generate = GenerateDefaults::from_json(g)?;
        } else {
            // scorciatoia: parametri di generazione direttamente alla radice
            c.generate = GenerateDefaults::from_json(j)?;
        }
        Ok(c)
    }

    fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_assoluti() {
        let d = GenerateDefaults::default();
        assert_eq!(d.temperature, Some(0.0)); // greedy: deterministico, scelta F2
        assert_eq!(d.max_tokens, Some(256));
        assert_eq!(d.top_p, None);
    }

    #[test]
    fn merge_parziale_vince_dove_specifica() {
        let base = GenerateDefaults::default();
        let file = GenerateDefaults {
            temperature: Some(0.7),
            top_p: None,
            top_k: Some(40),
            max_tokens: None,
        };
        let fuso = base.merge(&file);
        assert_eq!(fuso.temperature, Some(0.7)); // il file ha vinto
        assert_eq!(fuso.max_tokens, Some(256)); // ereditato dal default
        assert_eq!(fuso.top_k, Some(40));
        assert_eq!(fuso.top_p, None);
    }

    #[test]
    fn catena_tre_strati() {
        let d = GenerateDefaults::default();
        let file = GenerateDefaults {
            temperature: Some(0.7), top_p: None, top_k: None, max_tokens: Some(512),
        };
        let cli = GenerateDefaults {
            temperature: None, top_p: Some(0.9), top_k: None, max_tokens: None,
        };
        let fuso = d.merge(&file).merge(&cli);
        assert_eq!(fuso.temperature, Some(0.7)); // cli non lo tocca → resta del file
        assert_eq!(fuso.max_tokens, Some(512));
        assert_eq!(fuso.top_p, Some(0.9));
    }

    #[test]
    fn file_completo() {
        let src = r#"{
            "server": {"host": "0.0.0.0", "port": 9000},
            "model": {"path": "/tmp/m.gguf"},
            "generate": {"temperature": 0.8, "top_k": 20}
        }"#;
        let c = FileConfig::parse(src).unwrap();
        assert_eq!(c.server.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(c.server.port, Some(9000));
        assert_eq!(c.model_path.as_deref(), Some("/tmp/m.gguf"));
        assert_eq!(c.generate.temperature, Some(0.8));
        assert_eq!(c.generate.max_tokens, None); // non toccato
    }

    #[test]
    fn scorciatoia_radice() {
        let c = FileConfig::parse(r#"{"temperature": 1.1}"#).unwrap();
        assert_eq!(c.generate.temperature, Some(1.1));
        assert_eq!(c.server.port, None);
    }

    #[test]
    fn errori_chiariti() {
        assert!(FileConfig::parse("{").is_err());
        assert!(FileConfig::parse(r#"{"generate":{"temperature":"calda"}}"#).is_err());
        assert!(FileConfig::parse(r#"{"generate":{"temperature":99}}"#).is_err());
        assert!(FileConfig::parse(r#"{"server":{"port":99999}}"#).is_err());
        assert!(FileConfig::parse("[1,2]").is_err());
    }

    #[test]
    fn null_in_top_p_e_top_k_ignorato() {
        // esplicitare null = "non toccare" (utile nelle patch)
        let g = GenerateDefaults::patch_from_str(r#"{"top_p":null,"temperature":0.5}"#).unwrap();
        assert_eq!(g.temperature, Some(0.5));
        assert_eq!(g.top_p, None);
    }

    #[test]
    fn patch_runtime() {
        let g = GenerateDefaults::patch_from_str(r#"{"max_tokens": 42}"#).unwrap();
        assert_eq!(g.max_tokens, Some(42));
        assert_eq!(g.temperature, None); // patch parziale
    }
}
