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

#[test]
#[ignore = "richiede il GGUF ufficiale da ~1.1GB (vedi T002)"]
fn lfm3_pesi_forme() {
    use s_models::lfm3::{LayerKind, Lfm3Weights};

    let path = gguf_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: file GGUF assente ({path}) — scaricarlo come da T002");
        return;
    }

    let gguf = s_gguf::GgufFile::open(&path)
        .unwrap_or_else(|e| panic!("open GGUF {path}: {e}"));
    let cfg = s_models::lfm3::Config::from_gguf(&gguf)
        .unwrap_or_else(|e| panic!("Config::from_gguf: {e}"));

    // T003: carica TUTTI i 24 layer verificando ogni forma contro l'header.
    let w = Lfm3Weights::load(&gguf, &cfg)
        .unwrap_or_else(|e| panic!("Lfm3Weights::load: {e}"));

    assert_eq!(w.layers.len(), 24, "layer caricati");
    let n_short = w.layers.iter().filter(|l| l.kind == LayerKind::ShortConv).count();
    let n_attn = w.layers
        .iter()
        .filter(|l| l.kind == LayerKind::SlidingAttn || l.kind == LayerKind::FullAttn)
        .count();
    assert_eq!(n_short, 13, "shortconv attesi 13");
    assert_eq!(n_attn, 11, "attention attesi 11");

    // i 5 sliding sono esattamente gli indici 4, 8, 12, 16, 20
    assert_eq!(w.sliding_indices(), vec![4, 8, 12, 16, 20], "sliding");
    assert_eq!(w.kind_counts(), (13, 11, 5), "kind_counts pesi");

    // coerenza pesi vs config: ogni layer ha il kind della config
    for (li, l) in w.layers.iter().enumerate() {
        let k = cfg.layer_kind(li).unwrap_or_else(|e| panic!("layer_kind {li}: {e}"));
        assert_eq!(l.kind, k, "kind layer {li}");
    }

    // spot-check: una forma per famiglia contro l'header grezzo
    let conv0 = gguf.tensor("blk.0.shortconv.conv.weight").expect("blk.0 conv");
    assert_eq!(conv0.dims, vec![7, 2048], "blk.0 conv dims");
    let q2 = gguf.tensor("blk.2.attn_q.weight").expect("blk.2 q");
    assert_eq!(q2.dims, vec![2048, 2048], "blk.2 q dims");
    let g2 = gguf.tensor("blk.2.attn_gate.weight").expect("blk.2 gate");
    assert_eq!(g2.dims, vec![2048, 16], "blk.2 gate dims (per testa)");
    assert_eq!(w.token_embd.dims, vec![2048, 128000], "token_embd dims");
    assert_eq!(w.rope_freqs.dims, vec![64], "rope_freqs dims");

    println!(
        "✅ LFM3 PESI: 24 layer caricati (13 shortconv + 11 attention, sliding {:?}), \
         287 forme verificate, 0 correzioni",
        w.sliding_indices(),
    );
}


