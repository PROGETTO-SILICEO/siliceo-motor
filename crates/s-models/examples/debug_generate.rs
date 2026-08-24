//! Debug: generazione greedy per validare il forward.
use s_models::Model;

fn main() {
    let path = std::env::args().nth(1).expect("usage: debug_generate <gguf> [prompt]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "La capitale dell'Italia è".into());

    let tok = s_tokenizer::Tokenizer::from_gguf(&path).unwrap();
    let model = Model::load(&path).unwrap();

    let mut ids = tok.encode(&prompt).unwrap();
    eprintln!("IDS={:?}", ids);
    println!("prompt: {prompt:?} -> {:?} ids", ids.len());

    for _ in 0..(std::env::args().nth(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(12)) {
        let logits = model.forward(&ids);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        ids.push(argmax);
        let piece = tok.decode(&[argmax]).unwrap_or("?".into());
        print!("{piece:?} (id {argmax}) ");
        if argmax == tok.eos_id().unwrap_or(u32::MAX) {
            break;
        }
    }
    println!();
}
// stampa anche gli id del prompt su stderr
