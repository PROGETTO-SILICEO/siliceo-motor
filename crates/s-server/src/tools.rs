//! Tool calling — AC-2 (docs/AGENTIC_CORE.md).
//!
//! Formato: convenzione ecosystem hermes/Qwen2.5 — le firme degli strumenti
//! vengono presentate al modello in un blocco <tools> nelle istruzioni di
//! sistema, e le chiamate attese nell'output dentro tag <tool_call>:
//!
//!   <tool_call>
//!   {"name": "meteo", "arguments": {"citta": "Bergamo"}}
//!   </tool_call>
//!
//! Il LOOP AGENTIC resta lato client: il server rileva le chiamate, le
//! restituisce nel formato OpenAI (`message.tool_calls`, finish_reason
//! "tool_calls") e accetta i risultati come messaggi role "tool".
//! Non esegue MAI azioni: è un motore di inferenza.

use s_config::json::{parse, Json};

/// Definizione di uno strumento dichiarato dal client.
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// Schema dei parametri (JSON Schema), serializzato com'è.
    pub parameters: Option<String>,
}

/// Estrae le definizioni dal campo `tools` della richiesta.
/// Forma attesa: [{"type":"function","function":{"name":...,"description":...,"parameters":...}}]
pub fn parse_tools(j: Option<&Json>) -> Result<Vec<ToolDef>, String> {
    let arr = match j {
        None => return Ok(Vec::new()),
        Some(v) => v.as_arr().ok_or("tools deve essere un array")?,
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let f = t
            .get("function")
            .ok_or("tool senza 'function'")?;
        let name = f
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or("tool.function senza 'name' stringa")?
            .to_string();
        if name.is_empty() {
            return Err("tool.function.name vuoto".into());
        }
        let description =
            f.get("description").and_then(|d| d.as_str()).unwrap_or("").to_string();
        let parameters = f.get("parameters").map(|p| p.dump());
        out.push(ToolDef { name, description, parameters });
    }
    Ok(out)
}

/// Sezione istruzioni strumenti da anteporre alla conversazione.
/// Stile hermes/Qwen2.5: i modelli addestrati con questo formato lo riconoscono.
pub fn tools_section(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "# Tools\n\n\
         You can call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n",
    );
    for t in tools {
        let mut def = String::from("{\"type\": \"function\", \"function\": {\"name\": ");
        super::write_json_str(&mut def, &t.name);
        def.push_str(", \"description\": ");
        super::write_json_str(&mut def, &t.description);
        match &t.parameters {
            Some(p) => {
                def.push_str(", \"parameters\": ");
                def.push_str(p);
            }
            None => def.push_str(", \"parameters\": {}"),
        }
        def.push_str("}}");
        s.push_str(&def);
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\n\
         When you need to call a function, write the call inside \
         <tool_call></tool_call> tags like this:\n\
         <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\
         After the function results are provided as a \"tool\" message, continue \
         answering the user. If there is nothing to call, just answer normally.",
    );
    s
}

/// Una chiamata rilevata nell'output del modello.
pub struct DetectedCall {
    pub name: String,
    /// Argomenti come STRINGA JSON (formato OpenAI: arguments è una stringa).
    pub arguments: String,
}

/// Risultato dell'analisi dell'output: contenuto pulito + chiamate trovate.
pub struct ToolScan {
    /// Testo fuori dai blocchi <tool_call>, trimmato.
    pub content: String,
    pub calls: Vec<DetectedCall>,
}

impl ToolScan {
    /// true se è stata trovata almeno una chiamata valida.
    pub fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }
}

/// Analizza l'output del modello cercando blocchi <tool_call>{...}</tool_call>.
///
// Robustezza documentata:
// - il JSON interno viene VALIDATO col nostro parser; se non parsea il blocco
//   resta nel content (il modello ha sbagliato: non lo nascondiamo);
// - "arguments" può essere oggetto o stringa JSON: entrambi normalizzati a stringa;
// - i tag accettano whitespace opzionale intorno al JSON.
pub fn scan(text: &str) -> ToolScan {
    let mut calls = Vec::new();
    let mut clean = String::new();
    let mut rest = text;

    while let Some(start) = rest.find("<tool_call>") {
        clean.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        match after.find("</tool_call>") {
            Some(end) => {
                let inner = after[..end].trim();
                match parse_call(inner) {
                    Ok(call) => calls.push(call),
                    Err(_) => {
                        // blocco non parsabile: lo riemettiamo nel content
                        clean.push_str(&rest[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
                    }
                }
                rest = &after[end + "</tool_call>".len()..];
            }
            None => {
                // tag aperto senza chiusura: testo normale (modello troncato)
                clean.push_str(&rest[start..]);
                rest = "";
            }
        }
    }
    clean.push_str(rest);

    ToolScan { content: clean.trim().to_string(), calls }
}

fn parse_call(inner: &str) -> Result<DetectedCall, String> {
    let j = parse(inner)?;
    let name = j
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or("tool_call senza 'name'")?
        .to_string();
    let arguments = match j.get("arguments") {
        Some(Json::Obj(_)) | Some(Json::Arr(_)) => j.get("arguments").unwrap().dump(),
        Some(Json::Str(s)) => s.clone(),
        _ => "{}".to_string(),
    };
    Ok(DetectedCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use s_config::json::parse;

    const TOOLS_JSON: &str = r#"[
      {"type":"function","function":{
         "name":"meteo",
         "description":"Meteo di una citta",
         "parameters":{"type":"object","properties":{"citta":{"type":"string"}},"required":["citta"]}
      }},
      {"type":"function","function":{
         "name":"ora",
         "description":"Ora corrente"
      }}
    ]"#;

    #[test]
    fn parsing_dichiarazione_tools() {
        let tools = parse_tools(Some(&parse(TOOLS_JSON).unwrap())).unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "meteo");
        assert!(tools[0].parameters.as_deref().unwrap().contains("\"citta\""));
        assert_eq!(tools[1].parameters, None);
        assert!(parse_tools(None).unwrap().is_empty());
    }

    #[test]
    fn tools_vuoti_nessuna_sezione() {
        assert_eq!(tools_section(&[]), "");
    }

    #[test]
    fn sezione_contiene_firme_e_istruzioni() {
        let tools = parse_tools(Some(&parse(TOOLS_JSON).unwrap())).unwrap();
        let s = tools_section(&tools);
        assert!(s.contains("<tools>"));
        assert!(s.contains("\"meteo\""));
        assert!(s.contains("<tool_call>"));
        // la descrizione con caratteri speciali deve essere JSON-escaped
        assert!(!s.contains("Meteo di una cita\"")); // no quote non escaped
    }

    #[test]
    fn scansione_rileva_chiamata_con_oggetto_args() {
        let scan = scan("Ecco che ti serve.\n<tool_call>\n{\"name\": \"meteo\", \"arguments\": {\"citta\": \"Bergamo\"}}\n</tool_call>\nfatto.");
        assert_eq!(scan.calls.len(), 1);
        assert_eq!(scan.calls[0].name, "meteo");
        assert_eq!(scan.calls[0].arguments, "{\"citta\":\"Bergamo\"}");
        assert_eq!(scan.content, "Ecco che ti serve.\n\nfatto.");
        assert!(scan.has_calls());
    }

    #[test]
    fn scansione_args_stringa_e_multipli_blocchi() {
        let scan = scan(
            "<tool_call>{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":1}\"}</tool_call>\
             <tool_call>{\"name\":\"b\"}</tool_call>",
        );
        assert_eq!(scan.calls.len(), 2);
        assert_eq!(scan.calls[0].name, "a");
        assert_eq!(scan.calls[0].arguments, "{\"x\":1}"); // normalizzata a oggetto
        assert_eq!(scan.calls[1].arguments, "{}"); // assenti → oggetto vuoto
        assert_eq!(scan.content, "");
    }

    #[test]
    fn blocco_non_parsabile_restane_nel_content() {
        let txt = "<tool_call>{nome: senza virgolette}</tool_call>";
        let scan = scan(txt);
        assert!(scan.calls.is_empty());
        assert_eq!(scan.content, txt); // riemesso, non nascosto
    }

    #[test]
    fn tag_aperto_senza_chiusura_testo_normale() {
        let scan = scan("vorrei <tool_call> ma poi tronca");
        assert!(scan.calls.is_empty());
        assert_eq!(scan.content, "vorrei <tool_call> ma poi tronca");
    }

    #[test]
    fn nessun_tool_output_pulito() {
        let scan = scan("Risposta normale, nessuna chiamata.");
        assert!(!scan.has_calls());
        assert_eq!(scan.content, "Risposta normale, nessuna chiamata.");
    }
}
