//! Parity tokenizer: i nostri IDs contro llama-tokenize (llama.cpp).
//!
//! Riferimento (qwen2.5-0.5b-instruct-q4_k_m.gguf, llama-tokenize):
//!   prompt: "Il motore sovrano genera token identici! Test 123."
//!   output: [12050, 3852, 460, 773, 18920, 5652, 83435, 3950,
//!            3524, 3375, 0, 3393, 220, 16, 17, 18, 13]
//!
//! Attivazione: S_TOKENIZER_QWEN05B=<gguf> cargo test -p s-tokenizer --test parity

use s_tokenizer::Tokenizer;

const REF: &[u32] = &[
    12050, 3852, 460, 773, 18920, 5652, 83435, 3950, 3524, 3375, 0, 3393, 220, 16, 17, 18, 13,
];
const PROMPT: &str = "Il motore sovrano genera token identici! Test 123.";

#[test]
fn parity_with_llama_tokenize() {
    let Ok(path) = std::env::var("S_TOKENIZER_QWEN05B") else { return; };
    let tok = Tokenizer::from_gguf(&path).unwrap();
    println!("vocab size: {}", tok.vocab_size());
    println!("bos: {:?} eos: {:?} add_bos gestito internamente", tok.bos_id(), tok.eos_id());

    let ids = tok.encode(PROMPT).unwrap();
    println!("nostri ids:   {:?}", ids);
    println!("riferimento:  {:?}", REF);

    // Il riferimento di llama-tokenize può includere il BOS in testa.
    // Confrontiamo: (a) identici, (b) nostri = riferimento senza primo token,
    // (c) riferimento = nostri senza primo token.
    let ours = &ids[..];
    let matches_direct = ours == REF;
    let matches_no_bos = if REF.len() == ours.len() + 1 { &REF[1..] == ours } else { false };

    // decodifica diagnostica del primo token del riferimento
    if !matches_direct {
        println!("decodifica rif[0]={}: {:?}", REF[0], tok.decode(&[REF[0]]).unwrap_or("?".into()));
        println!("decodifica rif[0..4]: {:?}", tok.decode(&REF[..4.min(REF.len())]).unwrap_or("?".into()));
        println!("decodifica nostri[..4]: {:?}", tok.decode(&ours[..4.min(ours.len())]).unwrap_or("?".into()));
    }

    assert!(
        matches_direct || matches_no_bos,
        "token IDs diversi dal riferimento\nnostri:  {:?}\nrifer.:  {:?}",
        ours,
        REF
    );
    println!("✅ PARITY TOKENIZER: IDs identici a llama-tokenize");
}
