//! Audit STRUTTURALE dei tensori di LFM3-1.7B — T002, riverifica indipendente.
//!
//! Non si limita a contare 287: **deriva** il numero dalla struttura e poi
//! verifica che il file contenga esattamente quei tensori, senza mancanti né
//! extra. Un conteggio può coincidere per caso; una partizione no.
//!
//! Struttura attesa (misurata sull'header il 27/10/2026):
//!   - 13 layer shortconv, 10 tensori ciascuno → 130
//!   - 11 layer attention, 14 tensori ciascuno → 154
//!   - 3 tensori globali → 3
//!   totale derivato = 287 = `header.tensor_count` dichiarato dal file
//!
//! Il test degrada a SKIP se il GGUF non è in cache (niente verdi finti).

use s_gguf::GgufFile;
use std::collections::{BTreeMap, BTreeSet};

const DEFAULT_PATH: &str = "/home/guardiano/.cache/siliceo-models/LFM3-1.7B-Q4_K_M.gguf";

/// Tensori per layer shortconv (13 layer): norma pre, 2 norm post, ffn, conv.
const SHORTCONV_SUFFIXES: [&str; 10] = [
    "attn_norm.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_norm.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
    "shortconv.conv.weight",
    "shortconv.in_proj.weight",
    "shortconv.out_proj.weight",
];

/// Tensori per layer attention (11 layer): q/k/v/o + gate + 2 qk-norm + 4 norm + ffn.
const ATTN_SUFFIXES: [&str; 14] = [
    "attn_gate.weight",
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_norm.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
];

const GLOBALS: [&str; 3] = ["output_norm.weight", "rope_freqs.weight", "token_embd.weight"];

#[test]
fn lfm3_tensori_derivati_dalla_struttura() {
    let path =
        std::env::var("S_TOKENIZER_LFM3_GGUF").unwrap_or_else(|_| DEFAULT_PATH.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: GGUF LFM3 non presente in {path}");
        return;
    }
    let g = GgufFile::open(&path).expect("GGUF apribile");

    // 1. il file dichiara il proprio conteggio nell'header
    assert_eq!(
        g.header.tensor_count, 287,
        "header.tensor_count inatteso: {}",
        g.header.tensor_count
    );
    assert_eq!(
        g.tensors.len() as u64,
        g.header.tensor_count,
        "tensori letti ({}) != dichiarati ({})",
        g.tensors.len(),
        g.header.tensor_count
    );

    // 2. partizione per layer
    let mut globali: BTreeSet<String> = BTreeSet::new();
    let mut per_layer: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for t in &g.tensors {
        match t.name.strip_prefix("blk.") {
            Some(rest) => {
                let (idx, suffix) = rest
                    .split_once('.')
                    .unwrap_or_else(|| panic!("nome layer malformato: {}", t.name));
                let i: usize = idx.parse().unwrap_or_else(|_| panic!("indice non numerico: {}", t.name));
                per_layer.entry(i).or_default().insert(suffix.to_string());
            }
            None => {
                globali.insert(t.name.clone());
            }
        }
    }

    assert_eq!(globali.len(), 3, "globali inattesi: {globali:?}");
    assert_eq!(
        globali,
        GLOBALS.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>(),
        "set dei tensori globali diverso dall'atteso"
    );
    assert_eq!(per_layer.len(), 24, "layer trovati: {}", per_layer.len());

    // 3. classificazione e verifica ESATTA del set per ogni layer
    let mut n_shortconv = 0;
    let mut n_attn = 0;
    let mut sliding_candidates: Vec<usize> = Vec::new();
    for (i, set) in &per_layer {
        let has_conv = set.contains("shortconv.conv.weight");
        let has_attn = set.contains("attn_q.weight");
        assert!(
            has_conv ^ has_attn,
            "layer {i} né shortconv né attention: {set:?}"
        );

        let atteso: BTreeSet<String> = if has_conv {
            n_shortconv += 1;
            SHORTCONV_SUFFIXES.iter().map(|s| s.to_string()).collect()
        } else {
            n_attn += 1;
            sliding_candidates.push(*i);
            ATTN_SUFFIXES.iter().map(|s| s.to_string()).collect()
        };
        assert_eq!(
            *set, atteso,
            "layer {i}: set di tensori diverso dall'atteso\n mancanti: {:?}\n extra: {:?}",
            atteso.difference(set).collect::<Vec<_>>(),
            set.difference(&atteso).collect::<Vec<_>>()
        );
    }

    // 4. la derivazione aritmetica deve coincidere col dichiarato
    assert_eq!(n_shortconv, 13, "layer shortconv: {n_shortconv}");
    assert_eq!(n_attn, 11, "layer attention: {n_attn}");
    let derivato = n_shortconv * SHORTCONV_SUFFIXES.len()
        + n_attn * ATTN_SUFFIXES.len()
        + GLOBALS.len();
    assert_eq!(derivato, 287, "derivazione strutturale: {derivato}");
    assert_eq!(derivato as u64, g.header.tensor_count);

    // 5. gli indici attention devono essere quelli misurati
    assert_eq!(
        sliding_candidates,
        vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        "indici attention inattesi"
    );

    eprintln!(
        "✅ LFM3 audit: 13×10 + 11×14 + 3 = {derivato} tensori, header dichiara {}, partizione esatta",
        g.header.tensor_count
    );
}
