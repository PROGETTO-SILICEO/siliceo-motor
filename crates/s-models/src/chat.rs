//! Template di chat dinamici — D6d.
//!
//! Il prompt di chat NON è più cablato: ogni modello caricato porta con sé
//! il proprio formato, rilevato in quest'ordine (il primo che vale):
//! 1. override dalla config (`model.template`)
//! 2. campo `tokenizer.chat_template` nel GGUF (riconoscimento per sottostringa,
//!    NON rendering Jinja completo)
//! 3. fallback per architettura
//!
//! Formati implementati:
//! - ChatML  (Qwen2/Qwen2.5, SmolLM2): `<|im_start|>{role}\n{content}<|im_end|>`
//! - Llama3: `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`

/// Formati di chat noti al motore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    ChatML,
    Llama3,
}

impl Format {
    pub fn name(&self) -> &'static str {
        match self {
            Format::ChatML => "chatml",
            Format::Llama3 => "llama3",
        }
    }

    /// Token che chiudono il turno dell'assistente (diventano extra EOS).
    pub fn stop_tokens(&self) -> &'static [&'static str] {
        match self {
            Format::ChatML => &["<|im_end|>"],
            Format::Llama3 => &["<|eot_id|>", "<|end_of_text|>"],
        }
    }
}

/// Il template di chat del modello caricato + da dove è stato ricavato.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    pub format: Format,
    /// "config" | "gguf" | "arch" | "default"
    pub source: &'static str,
}

impl ChatTemplate {
    /// Rileva il formato. Ordine: config > gguf > arch > default.
    ///
    /// * `arch`: valore del KV `general.architecture` ("qwen2", "llama", ...)
    /// * `gguf_tpl`: contenuto del KV `tokenizer.chat_template` se stringa
    /// * `over`: override dalla config (None = auto)
    pub fn detect(
        arch: &str,
        gguf_tpl: Option<&str>,
        over: Option<&str>,
    ) -> Result<Self, String> {
        if let Some(o) = over {
            let format = match o {
                "auto" => return Self::detect(arch, gguf_tpl, None),
                "chatml" => Format::ChatML,
                "llama3" => Format::Llama3,
                other => {
                    return Err(format!(
                        "template '{other}' non noto (validi: auto, chatml, llama3)"
                    ))
                }
            };
            return Ok(Self { format, source: "config" });
        }

        if let Some(t) = gguf_tpl {
            // riconoscimento per firma: il Jinja completo non è un obiettivo ora
            if t.contains("<|im_start|>") {
                return Ok(Self { format: Format::ChatML, source: "gguf" });
            }
            if t.contains("<|start_header_id|>") {
                return Ok(Self { format: Format::Llama3, source: "gguf" });
            }
        }

        match arch {
            "qwen2" => Ok(Self { format: Format::ChatML, source: "arch" }),
            // GUESS documentato: l'arch "llama" copre llama2 e llama3; senza
            // template incorporato non si distinguono. La maggior parte dei
            // modelli moderni ha il template nel GGUF → questo ramo è raro.
            "llama" => Ok(Self { format: Format::Llama3, source: "arch" }),
            other => {
                let _ = other;
                Ok(Self { format: Format::ChatML, source: "default" })
            }
        }
    }

    /// Rende il prompt completo, pronto per la tokenizzazione.
    /// `messages`: coppie (ruolo, contenuto) nell'ordine della conversazione.
    pub fn render(&self, messages: &[(&str, &str)]) -> String {
        match self.format {
            Format::ChatML => {
                let mut out = String::new();
                for (role, content) in messages {
                    out.push_str("<|im_start|>");
                    out.push_str(role);
                    out.push('\n');
                    out.push_str(content);
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
            Format::Llama3 => {
                let mut out = String::from("<|begin_of_text|>");
                for (role, content) in messages {
                    out.push_str("<|start_header_id|>");
                    out.push_str(role);
                    out.push_str("<|end_header_id|>\n\n");
                    out.push_str(content);
                    out.push_str("<|eot_id|>");
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TPL_QWEN: &str =
        r#"{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"#;
    const TPL_LLAMA3: &str =
        r#"{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}"#;

    #[test]
    fn chatml_render_uguale_al_vecchio_cablato() {
        // NON-regressione: il render ChatML produce ESATTAMENTE ciò che
        // render_chatml produceva prima di D6d (verificato in parity F2).
        let t = ChatTemplate { format: Format::ChatML, source: "test" };
        let out = t.render(&[("system", "Sei utile."), ("user", "Ciao")]);
        assert_eq!(out, "<|im_start|>system\nSei utile.<|im_end|>\n<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn llama3_render() {
        let t = ChatTemplate { format: Format::Llama3, source: "test" };
        let out = t.render(&[("user", "Hi"), ("assistant", "Hello!")]);
        assert_eq!(out, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    #[test]
    fn rilevamento_da_gguf() {
        let t = ChatTemplate::detect("qwen2", Some(TPL_QWEN), None).unwrap();
        assert_eq!(t.format, Format::ChatML);
        assert_eq!(t.source, "gguf");
        let t = ChatTemplate::detect("llama", Some(TPL_LLAMA3), None).unwrap();
        assert_eq!(t.format, Format::Llama3);
        assert_eq!(t.source, "gguf");
    }

    #[test]
    fn override_config_vince_su_tutto() {
        let t = ChatTemplate::detect("qwen2", Some(TPL_QWEN), Some("llama3")).unwrap();
        assert_eq!(t.format, Format::Llama3);
        assert_eq!(t.source, "config");
        // 'auto' = torna alla catena normale
        let t = ChatTemplate::detect("qwen2", Some(TPL_QWEN), Some("auto")).unwrap();
        assert_eq!(t.format, Format::ChatML);
        assert_eq!(t.source, "gguf");
    }

    #[test]
    fn override_sconosciuto_errore_chiaro() {
        let e = ChatTemplate::detect("qwen2", None, Some("mystyle")).unwrap_err();
        assert!(e.contains("mystyle"));
    }

    #[test]
    fn fallback_per_architettura_e_default() {
        let t = ChatTemplate::detect("qwen2", None, None).unwrap();
        assert_eq!((t.format, t.source), (Format::ChatML, "arch"));
        let t = ChatTemplate::detect("llama", None, None).unwrap();
        assert_eq!((t.format, t.source), (Format::Llama3, "arch"));
        let t = ChatTemplate::detect("gemma2", None, None).unwrap();
        assert_eq!((t.format, t.source), (Format::ChatML, "default"));
    }

    #[test]
    fn template_gguf_non_riconoscibile_cade_sull_arch() {
        // template Jinja esoterico senza firme note → fallback, NON errore
        let t = ChatTemplate::detect("qwen2", Some("{{ bos }}{{ content }}"), None).unwrap();
        assert_eq!((t.format, t.source), (Format::ChatML, "arch"));
    }

    #[test]
    fn stop_tokens_per_formato() {
        assert_eq!(Format::ChatML.stop_tokens(), &["<|im_end|>"]);
        assert!(Format::Llama3.stop_tokens().contains(&"<|eot_id|>"));
    }
}
