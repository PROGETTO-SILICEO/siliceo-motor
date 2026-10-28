//! Pre-tokenizer `lfm2` (LFM3) — T001.
//!
//! Provenienza dei valori attesi (test `split_matches_hf_lfm3`):
//! tokenizer.json di `LiquidAI/LFM3-1.7B` (HF hub, in cache locale),
//! `pre_tokenizer = Sequence(Split(Regex=<GPT-2 family>), ByteLevel)`.
//! Misurato con `tokenizers==0.23.1`:
//!   `Tokenizer.from_file(...).pre_tokenizer.pre_tokenize_str(prompt)`
//! e riportato a segmenti grezzi (ByteLevel invertito: `Ġ`→spazio,
//! `Ċ`→`\n`, `Ã¨`→`è`, resto identico perché ASCII stampabile).
//! La regex Split dell'originale è carattere-per-carattere quella usata
//! da `PreTokenizer::for_kind("lfm2")` (famiglia GPT-2) — il test
//! `upstream_regex_invariato` lo ricontrolla contro il file originale
//! quando disponibile.
//!
//! RIVERIFICATO in modo indipendente il 28/10/2026: le tre sequenze attese qui
//! sotto sono state ricalcolate da zero con `tokenizers` sul `tokenizer.json`
//! in cache, invertendo ByteLevel con la tabella GPT-2 (`Ġ`→spazio, `Ċ`→`\n`,
//! byte UTF-8 → carattere). Coincidono carattere per carattere: il test NON è
//! circolare, non è stato adattato all'output del nostro codice.

use s_tokenizer::pre::PreTokenizer;

#[test]
fn lfm2_kind_ok() {
    assert!(
        PreTokenizer::for_kind("lfm2").is_ok(),
        "for_kind(\"lfm2\") deve restituire Ok (GGUF LFM3 dichiara tokenizer.ggml.pre = lfm2)"
    );
}

#[test]
fn lfm2_stessa_famiglia_di_gpt2() {
    // Ipotesi del task: lfm2 = stessa regex GPT-2. Lo fissiamo: se un giorno
    // le due famiglie divergono, questo test lo segnala.
    let probe = "l'modello risponde: 42 è la risposta? Sì!\nNuova riga. Don't stop... ok?";
    let a = PreTokenizer::for_kind("lfm2").unwrap();
    let b = PreTokenizer::for_kind("gpt2").unwrap();
    assert_eq!(a.split(probe), b.split(probe), "lfm2 e gpt2 devono splittare identico");
}

#[test]
fn split_matches_hf_lfm3() {
    let pre = PreTokenizer::for_kind("lfm2").unwrap();

    // Prompt 1 — stesso del test di parity qwen2 (confronto diretto possibile).
    // HF pre_tokenize_str:
    //   Il | Ġmotore | Ġsovrano | Ġgenera | Ġtoken | Ġidentici | ! | ĠTest | Ġ | 123 | .
    let s1 = pre.split("Il motore sovrano genera token identici! Test 123.");
    assert_eq!(
        s1,
        vec![
            "Il", " motore", " sovrano", " genera", " token", " identici", "!", " Test", " ",
            "123", "."
        ],
        "split lfm2 ≠ pre-tokenizer HF di LiquidAI/LFM3-1.7B (prompt 1)"
    );

    // Prompt 2 — apostrofi, accenti, newline.
    // HF pre_tokenize_str:
    //   l | 'm | odello | Ġrisponde | : | Ġ | 42 | ĠÃ¨ | Ġla | Ġrisposta
    //   | ? | ĠSÃ¬ | !Ċ | Nuova | Ġriga | .
    let s2 = pre.split("l'modello risponde: 42 è la risposta? Sì!\nNuova riga.");
    assert_eq!(
        s2,
        vec![
            "l", "'m", "odello", " risponde", ":", " ", "42", " è", " la", " risposta", "?",
            " Sì", "!\n", "Nuova", " riga", "."
        ],
        "split lfm2 ≠ pre-tokenizer HF di LiquidAI/LFM3-1.7B (prompt 2)"
    );

    // Prompt 3 — inglese, punteggiatura multipla, contrazione.
    // HF pre_tokenize_str:
    //   Hello | , | Ġworld | ! | Ġ | 123 | Ġtesting | ... | Ġdon | 't | Ġstop | .
    let s3 = pre.split("Hello, world! 123 testing... don't stop.");
    assert_eq!(
        s3,
        vec![
            "Hello", ",", " world", "!", " ", "123", " testing", "...", " don", "'t", " stop",
            "."
        ],
        "split lfm2 ≠ pre-tokenizer HF di LiquidAI/LFM3-1.7B (prompt 3)"
    );
}

/// Ricontrolla contro il file originale che la regex Split di LFM3 sia ancora
/// quella della famiglia GPT-2 (nessuna nuova dipendenza: contiene-stringa).
/// Sorgente: `S_TOKENIZER_LFM3_JSON` (= tokenizer.json di LiquidAI/LFM3-1.7B),
/// altrimenti cache HF standard, altrimenti SALTATO (come gli altri test
/// che richiedono file esterni: dicono perché, non finto ok).
#[test]
fn upstream_regex_invariato() {
    let path = match std::env::var("S_TOKENIZER_LFM3_JSON").ok().filter(|p| !p.is_empty()) {
        Some(p) => p,
        None => match default_hf_tokenizer_json() {
            Some(p) => p,
            None => {
                eprintln!(
                    "SALTATO: imposta S_TOKENIZER_LFM3_JSON (tokenizer.json di \
                     LiquidAI/LFM3-1.7B) per ricontrollare la regex contro l'originale"
                );
                return;
            }
        },
    };
    let testo = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("non leggo {path}: {e}"));
    // Regex Split dentro tokenizer.json (con escaping JSON dei backslash).
    const SPLIT_REGEX_JSON: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"#;
    assert!(
        testo.contains(SPLIT_REGEX_JSON),
        "tokenizer.json di LFM3 non contiene più la regex Split della famiglia GPT-2: \
         rivalutare il mapping di for_kind(\"lfm2\")\nfile: {path}"
    );
}

/// Cerca tokenizer.json di LFM3-1.7B nella cache standard di huggingface_hub.
fn default_hf_tokenizer_json() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let base = format!("{home}/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots");
    let snap = std::fs::read_dir(&base).ok()?;
    for entry in snap.flatten() {
        let cand = entry.path().join("tokenizer.json");
        if cand.is_file() {
            return cand.to_str().map(str::to_string);
        }
    }
    None
}
