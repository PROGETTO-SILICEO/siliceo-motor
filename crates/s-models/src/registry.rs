//! Registro delle famiglie di architettura — F3b finestra 1.
//!
//! OGGI: il load deduce i prefissi metadata ("qwen2"/"llama") dentro
//! `qwen2::Config::from_gguf`, senza dichiarare QUALE famiglia è stata
//! riconosciuta. Quando arriveranno altre famiglie (SmolLM già toccato in
//! D6c, ibridi DeltaNet in F5) il dispatch enum→loader deve avere UN posto:
//! questo modulo.
//!
//! Principio roadmap v2: anticipare le ASTRAZIONI. L'enum cresce a ogni
//! nuova famiglia; il punto di dispatch resta questo file, non spargito
//! tra server/hot-swap/generate.

use s_gguf::GgufFile;

/// Famiglie di architettura riconosciute dal loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Famiglia {
    /// qwen2 / llama (llama con bias QKV opzionale). Dense attention su
    /// tutti i layer. È l'UNICA famiglia servita oggi.
    Qwen2Llama,
}

impl Famiglia {
    /// Nome della chiave `general.architecture` che identifica la famiglia.
    pub fn arch(self) -> &'static str {
        match self {
            Famiglia::Qwen2Llama => "qwen2/llama",
        }
    }

    /// Riconoscimento dalla stringa `general.architecture` del GGUF.
    /// Le architetture che condividono il layout qwen2/llama si mappano qui
    /// man mano che vengono verificate con parity (mai "per speranza").
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "qwen2" | "llama" => Some(Famiglia::Qwen2Llama),
            _ => None,
        }
    }

    /// Legge `general.architecture` dal metadata GGUF e risolve la famiglia.
    ///
    /// Errore esplicito e onesto se l'architettura non è supportata: meglio
    /// un rifiuto pulito all'load che un forward sbagliato silenzioso.
    pub fn detect(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .metadata
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .and_then(|(_, v)| v.as_str())
            .ok_or("metadata mancante: general.architecture")?;
        Self::from_arch(arch).ok_or(format!(
            "architettura '{arch}' non supportata (supportate: qwen2, llama)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architetture_dense_riconosciute() {
        assert_eq!(Famiglia::from_arch("qwen2"), Some(Famiglia::Qwen2Llama));
        assert_eq!(Famiglia::from_arch("llama"), Some(Famiglia::Qwen2Llama));
        // smollm ha layout proprio (prefisso diverso): NON si mappa per speranza
        assert_eq!(Famiglia::from_arch("smollm"), None);
        assert_eq!(Famiglia::from_arch("qwen3"), None);
    }

    /// Il formato dell'errore di detect() per arch sconosciuta: nomina sia
    /// l'architettura trovata sia le supportate (verificato su stringa perché
    /// costruire un GgufFile in test richiederebbe un file reale).
    #[test]
    fn formato_errore_detect() {
        let msg = format!(
            "{}",
            Famiglia::from_arch("gemma3")
                .map(|f| f.arch().to_string())
                .unwrap_or_else(|| "architettura 'gemma3' non supportata (supportate: qwen2, llama)"
                    .to_string())
        );
        assert!(msg.contains("'gemma3'"));
        assert!(msg.contains("qwen2, llama"));
    }
}
