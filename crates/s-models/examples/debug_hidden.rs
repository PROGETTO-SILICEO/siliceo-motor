use s_models::qwen35::Model;
fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/opencode/Qwen3.5-0.8B-F16.gguf".to_string());
    let model = Model::load(path).expect("load");
    let tokens: Vec<u32> = vec![9419];
    let per = model.debug_per_layer_hidden(&tokens);
    println!("per_layer len {}", per.len());
    for (i, h) in per.iter().enumerate() {
        let first3 = &h[..3.min(h.len())];
        let norm: f32 = h.iter().map(|v| v*v).sum::<f32>().sqrt();
        println!("rust layer {} first3 {:?} norm {:.2}", i, first3, norm);
    }
    let last = model.debug_last_hidden(&tokens);
    println!("last hidden first10 {:?}", &last[..10]);
    let logits = model.forward(&tokens);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let logprobs: Vec<f32> = exp.iter().map(|e| (e/sum).ln()).collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a,&b| logprobs[b].partial_cmp(&logprobs[a]).unwrap());
    println!("top5 Hello:");
    for i in 0..5 {
        let id = order[i];
        println!("  {} id {} logprob {:.4} logit {:.4}", i, id, logprobs[id], logits[id]);
    }
}
