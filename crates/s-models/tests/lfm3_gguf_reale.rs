//! T002 — LFM3 vero dentro `s-gguf` + `Config::from_gguf`.
//!
//! Eseguito contro il GGUF ufficiale Q4_K_M (~1.1 GB):
//!   ~/.cache/siliceo-models/LFM3-1.7B-Q4_K_M.gguf
//!   (oppure env `S_MODELS_LFM3_Q4K` per override del percorso)
//!
//! Il GGUF è la verità, il codice si adegua: se un numero non torna,
//! NON aggiustare il test — aggiusta il codice o dichiara l'errore.
//!
//! Lanciare con:
//!   cargo test -p s-models --features lfm3 -- --include-ignored

#![cfg(feature = "lfm3")]

fn gguf_path() -> String {
    std::env::var("S_MODELS_LFM3_Q4K").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/guardiano".into());
        format!("{home}/.cache/siliceo-models/LFM3-1.7B-Q4_K_M.gguf")
    })
}

#[test]
#[ignore = "richiede il GGUF ufficiale da ~1.1GB (vedi T002)"]
fn lfm3_gguf_reale() {
    let path = gguf_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: file GGUF assente ({path}) — scaricarlo come da T002");
        return;
    }

    let gguf = s_gguf::GgufFile::open(&path)
        .unwrap_or_else(|e| panic!("open GGUF {path}: {e}"));

    // 1. architettura dichiarata
    let arch = gguf
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .map(|(_, v)| v.clone());
    assert_eq!(
        arch.as_ref().and_then(|v| v.as_str()),
        Some("lfm3"),
        "general.architecture inattesa: {arch:?}"
    );

    // 2. Config dal metadata reale
    let cfg = s_models::lfm3::Config::from_gguf(&gguf)
        .unwrap_or_else(|e| panic!("Config::from_gguf: {e}"));

    assert_eq!(cfg.n_layers, 24, "n_layers");
    assert_eq!(cfg.n_embd, 2048, "n_embd");
    assert_eq!(cfg.n_ff, 7680, "n_ff");
    assert_eq!(cfg.n_heads, 16, "n_heads");
    assert_eq!(cfg.n_kv_heads, 4, "n_kv_heads");
    assert_eq!(cfg.head_dim, 128, "head_dim");
    assert_eq!(cfg.vocab_size, 128000, "vocab_size");
    assert_eq!(cfg.sliding_window, 256, "sliding_window");
    assert_eq!(cfg.conv_l_cache, 7, "conv_l_cache");
    assert!(
        (cfg.rope_theta - 1e6).abs() < 1.0,
        "rope_theta: atteso 1e6, trovato {}",
        cfg.rope_theta
    );
    assert!(
        (cfg.rope_theta_swa - 1e4).abs() < 1.0,
        "rope_theta_swa: atteso 1e4, trovato {}",
        cfg.rope_theta_swa
    );

    // 3. struttura tri-hybrid misurata
    assert_eq!(
        cfg.kind_counts().expect("kind_counts"),
        (13, 5, 6),
        "kind_counts (shortconv, sliding, full)"
    );

    // 4. numero di tensori letti dal GGUF
    assert_eq!(gguf.tensors.len(), 287, "tensori nel GGUF");

    println!(
        "✅ LFM3 REALE: arch=lfm3 layers={} embd={} ff={} heads={}/{:?} headdim={} vocab={} swa={} lcache={} rope={}/{} kinds={:?} tensori={}",
        cfg.n_layers,
        cfg.n_embd,
        cfg.n_ff,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.sliding_window,
        cfg.conv_l_cache,
        cfg.rope_theta,
        cfg.rope_theta_swa,
        cfg.kind_counts().unwrap(),
        gguf.tensors.len(),
    );
}
