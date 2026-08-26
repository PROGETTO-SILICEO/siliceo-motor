//! Sonda: SOLO Famiglia::detect sul file, niente tokenizer niente pesi.
fn main() {
    let path = std::env::args().nth(1).expect("uso: probe <file.gguf>");
    let t0 = std::time::Instant::now();
    match s_gguf::GgufFile::open(&path) {
        Ok(gguf) => {
            let fam = s_models::Famiglia::detect(&gguf);
            eprintln!("({:.2?}) detect: {:?}", t0.elapsed(), fam);
        }
        Err(e) => eprintln!("({:.2?}) apertura fallita: {e}", t0.elapsed()),
    }
}
