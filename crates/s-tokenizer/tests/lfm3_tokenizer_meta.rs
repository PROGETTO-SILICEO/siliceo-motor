//! Metadati del tokenizer LFM3 letti **dal file GGUF reale**.
//!
//! Chiude i due buchi che i report del turno di notte del 28/10/2026 hanno
//! dichiarato aperti:
//!
//! (a) `tokenizer.ggml.pre` era stato preso dal testo del task, non dal file.
//!     Qui si legge dal GGUF e si verifica che sia `lfm2`.
//! (b) `vocab_size = 128000` (da `token_embd.weight`) non era mai stato
//!     incrociato col tokenizer. Misurato il 28/10/2026 sul lato HF:
//!     `config.vocab_size = 128000`, `len(tokenizer) = 125033`, delta **2967**.
//!     Il delta NON è un artefatto della conversione GGUF: è padding già
//!     presente nel modello HF (Liquid ha addestrato con vocab paddato).
//!     Qui si fissa il lato GGUF: tokens nel file = 128000.
//!
//! Il test degrada a SKIP se il GGUF non è in cache (nessun fallimento finto).

use s_gguf::{GgufFile, GgufValue};

const DEFAULT_PATH: &str = "/home/guardiano/.cache/siliceo-models/LFM3-1.7B-Q4_K_M.gguf";

fn gguf_path() -> String {
    std::env::var("S_TOKENIZER_LFM3_GGUF").unwrap_or_else(|_| DEFAULT_PATH.to_string())
}

fn meta<'a>(g: &'a GgufFile, key: &str) -> Option<&'a GgufValue> {
    g.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[test]
fn lfm3_tokenizer_meta_dal_gguf() {
    let p = gguf_path();
    if !std::path::Path::new(&p).exists() {
        eprintln!("SKIP: GGUF LFM3 non presente in {p}");
        return;
    }
    let g = GgufFile::open(&p).expect("GGUF apribile");

    let arch = meta(&g, "general.architecture")
        .and_then(|v| v.as_str())
        .expect("general.architecture presente");
    assert_eq!(arch, "lfm3");

    // (a) il pre-tokenizer DICHIARATO DAL FILE
    let pre = meta(&g, "tokenizer.ggml.pre")
        .and_then(|v| v.as_str())
        .expect("tokenizer.ggml.pre presente");
    assert_eq!(pre, "lfm2", "pre-tokenizer inatteso: {pre}");

    let model = meta(&g, "tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .expect("tokenizer.ggml.model presente");
    assert_eq!(model, "gpt2", "modello tokenizer inatteso: {model}");

    // (b) dimensione del vocabolario nel GGUF
    let n_tokens = match meta(&g, "tokenizer.ggml.tokens").expect("tokenizer.ggml.tokens presente") {
        GgufValue::Array(v) => v.len(),
        other => panic!("tokenizer.ggml.tokens: atteso array, trovato {other:?}"),
    };
    assert_eq!(n_tokens, 128_000, "tokens nel GGUF inattesi: {n_tokens}");

    // Il padding è upstream, non nostro: 128000 - 125033 = 2967.
    // Se un giorno questo assert cambia, è un rilascio nuovo: riverificare il lato HF.
    assert_eq!(n_tokens - 125_033, 2_967);

    eprintln!(
        "✅ LFM3 tokenizer meta: arch={arch} model={model} pre={pre} tokens={n_tokens} (padding upstream {})",
        n_tokens - 125_033
    );
}
