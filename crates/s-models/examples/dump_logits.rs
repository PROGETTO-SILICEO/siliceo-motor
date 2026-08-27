use s_models::qwen35::Model;
use std::io::Write;
fn main() {
    let path = "/tmp/opencode/Qwen3.5-0.8B-F16.gguf";
    let model = Model::load(path).expect("load");
    let tokens = vec![9419u32];
    let logits = model.forward(&tokens);
    let mut f = std::fs::File::create("/tmp/rust_logits.bin").unwrap();
    for &v in &logits {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let logprobs: Vec<f32> = exp.iter().map(|e| (e/sum).ln()).collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a,&b| logprobs[b].partial_cmp(&logprobs[a]).unwrap());
    for i in 0..5 {
        println!("rust top{} id {} logit {:.4} logprob {:.4}", i, order[i], logits[order[i]], logprobs[order[i]]);
    }
    println!("rust logits saved {}", logits.len());
}
