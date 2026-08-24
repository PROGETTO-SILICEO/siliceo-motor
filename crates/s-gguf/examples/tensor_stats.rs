//! Statistiche complete di un tensore dequantizzato (confronto con gguf-py).
use s_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let name = std::env::args().nth(2).unwrap();
    let mut f = GgufFile::open(&path).unwrap();
    let d = f.tensor_data_f32(&name).unwrap();
    let ne = 896usize; // riga per attn_*
    let abs_max = d.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    println!("elementi={} |max|={:.6} media_ass={:.6}", d.len(), abs_max,
        d.iter().map(|v| v.abs()).sum::<f32>() / d.len() as f32);
    println!("riga0 primi4={:?}",
        d[0..4].iter().map(|v| (v*1e6).round()/1e6).collect::<Vec<_>>());
    if d.len() > 127 * ne + 4 {
        println!("riga127 primi4={:?}",
            d[127*ne..127*ne+4].iter().map(|v| (v*1e6).round()/1e6).collect::<Vec<_>>());
    }
}
