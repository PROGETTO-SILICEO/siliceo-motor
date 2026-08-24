//! Dump primi valori dequantizzati di un tensore (confronto con Python).
use s_gguf::GgufFile;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let name = std::env::args().nth(2).unwrap_or_else(|| "token_embd.weight".into());
    let skip = std::env::args().nth(3).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let mut f = GgufFile::open(&path).unwrap();
    // leggi solo il primo blocco: trucco — leggi tutto il tensore è pesante per 300MB,
    // quindi per il confronto leggiamo comunque tutto (0.5B ok in RAM)
    let d = f.tensor_data_f32(&name).unwrap();
    let start = skip as usize;
    println!("Rust primi 8:", );
    println!("{:?}", d[start..start+8].iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>());
}
