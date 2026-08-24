//! s-config — configurazione dinamica per siliceo-motor (D6).
//!
//! Config a strati, dal più debole al più forte (l'ultimo vince):
//! 1. DEFAULT   nel codice (`GenerateDefaults::default`, porta 8096, ...)
//! 2. SISTEMA   /etc/siliceo-motor/motor.json (opzionale)
//! 3. UTENTE    ./motor.json oppure --config PATH
//! 4. CLI       flag sulla riga di comando
//! 5. REQUEST   parametri nella singola richiesta (gestito da s-server)
//! 6. RUNTIME   POST /v1/config → patch a server acceso
//!
//! Ogni strato specifica SOLO ciò che vuole cambiare; il resto eredita.
//!
//! Zero dipendenze: il parser JSON è scritto in casa (modulo `json`).

pub mod config;
pub mod json;

pub use config::{FileConfig, GenerateDefaults, ServerDefaults};
pub use json::{parse, write_escaped, Json};
