//! Sanity dequant Q5_0: similarità coseno tra embedding di parole correlate.
use s_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let tok = s_tokenizer::Tokenizer::from_gguf(&path).unwrap();
    let mut f = GgufFile::open(&path).unwrap();
    let embd = f.tensor_data_f32("token_embd.weight").unwrap();
    let ne = f.tensor("token_embd.weight").unwrap().dims[0] as usize;

    let get = |word: &str| -> Vec<f32> {
        let ids = tok.encode(word).unwrap();
        let off = ids[0] as usize * ne;
        embd[off..off + ne].to_vec()
    };
    let cos = |a: &Vec<f32>, b: &Vec<f32>| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };

    let cane = get(" cane");
    let gatto = get(" gatto");
    let cavallo = get(" cavallo");
    let casa = get(" casa");
    let rand_vec: Vec<f32> = (0..ne).map(|i| embd[(i * 7919) % embd.len()]).collect();

    println!("cane~gatto   : {:.4} (atteso alto)", cos(&cane, &gatto));
    println!("cane~cavallo : {:.4} (atteso alto)", cos(&cane, &cavallo));
    println!("cane~casa    : {:.4} (atteso medio-basso)", cos(&cane, &casa));
    println!("cane~random  : {:.4} (atteso ~0)", cos(&cane, &rand_vec));
    println!("norma cane   : {:.4}", cane.iter().map(|x| x * x).sum::<f32>().sqrt());
}
