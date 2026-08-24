//! Parity logits F1: il nostro forward contro llama.cpp (beellama, nodo2070).
//!
//! Riferimento (llama-server /completion, n_probs=8, greedy):
//!   prompt: "Ciao dal motore sovrano! Test 123. Test 123."
//!   top-8 prossimi token (id, logprob):
//!     15 "0"      -2.3189
//!     4710 " \n\n" -2.8043
//!     20 "5"      -3.2001
//!     715 " \n"   -3.4199
//!     17 "2"      -3.5895
//!     16 "1"      -3.6796
//!
//! Token IDs del prompt (verificati identici da s-tokenizer):
//!   [34, 22516, 1674, 21470, 704, 11, 3815, 3852, 460, 773, 18920, 5652,
//!    342, 8832, 264, 220, 22, 20, 9628, 2687, 0, 3393, 220, 16, 17, 18, 13]
//!
//! Attivazione: S_MODELS_QWEN05B=<gguf> cargo test -p s-models --release --test parity_logits

#[test]
#[ignore = "lento in debug: lanciare con --release"]
fn parity_logits_with_llamacpp() {
    let Ok(path) = std::env::var("S_MODELS_QWEN05B") else { return; };

    let tokens: Vec<u32> = vec![
        34, 22516, 1674, 21470, 704, 11, 3815, 3852, 460, 773, 18920, 5652, 342, 8832, 264, 220,
        22, 20, 9628, 2687, 0, 3393, 220, 16, 17, 18, 13,
    ];

    let t0 = std::time::Instant::now();
    let model = s_models::Model::load(&path).unwrap_or_else(|e| panic!("load: {e}"));
    eprintln!("load in {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let logits = model.forward(&tokens);
    eprintln!("forward in {:?} ({} token)", t1.elapsed(), tokens.len());

    // logprobs nostri (softmax sui logits)
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    for e in exp.iter_mut() {
        *e /= sum;
    }
    let logprobs: Vec<f32> = exp.iter().map(|p| p.ln()).collect();

    // top-8 nostri
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logprobs[b].partial_cmp(&logprobs[a]).unwrap());

    // riferimento (id, logprob) da llama.cpp
    let reference: &[(u32, f32)] = &[
        (15, -2.3189),
        (4710, -2.8043),
        (20, -3.2001),
        (715, -3.4199),
        (17, -3.5895),
        (16, -3.6796),
    ];

    println!("top-8 nostri:");
    for (rank, &i) in order.iter().take(8).enumerate() {
        println!("  {rank}. id={} logprob={:.4}", i, logprobs[i]);
    }

    // confronto: gli id nel top-6 devono coincidere in ordine
    for (rank, &(ref_id, ref_lp)) in reference.iter().enumerate() {
        let ours_id = order[rank] as u32;
        let ours_idx = order[rank];
        assert_eq!(ours_id, ref_id, "rank {rank}: id {} != riferimento {ref_id}", ours_id);
        let diff = (logprobs[ours_idx] - ref_lp).abs();
        assert!(
            diff < 0.05,
            "rank {rank}: logprob {:.4} vs riferimento {ref_lp} (diff {diff:.4})",
            logprobs[ours_idx]
        );
    }
    println!("✅ PARITY LOGITS F1: top-6 identico a llama.cpp, logprob entro 0.05");
}
