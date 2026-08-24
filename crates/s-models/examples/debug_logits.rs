//! Debug: logits top-8 per un prompt, per parity con llama-server.
use s_models::Model;

fn main() {
    let path = std::env::args().nth(1).expect("usage: debug_logits <gguf> [prompt]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "C".into());

    let tok = s_tokenizer::Tokenizer::from_gguf(&path).unwrap();
    let model = Model::load(&path).unwrap();

    let ids = tok.encode(&prompt).unwrap();
    eprintln!("prompt: {prompt:?} -> {ids:?}");

    let logits = model.forward_traced(&ids);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    // softmax per confrontare i logprob di llama-server
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    for &i in idx.iter().take(8) {
        let lp = (logits[i] - max) - sum.ln();
        println!("id {i}: logprob {lp:.4}");
    }
}
