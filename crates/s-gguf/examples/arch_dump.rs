fn main() {
    let path = std::env::args().nth(1).unwrap();
    let f = s_gguf::GgufFile::open(&path).unwrap();
    for (k, v) in &f.metadata {
        if k.contains("attention") || k.contains("block_count") || k.contains("embedding")
            || k.contains("feed_forward") || k.contains("rope") || k.contains("rms") || k.contains("key_length") || k.contains("value_length") {
            println!("{k} = {v:?}");
        }
    }
    // tipi dei tensori del primo blocco
    for t in &f.tensors {
        if t.name.starts_with("blk.0.") || t.name == "output.weight" || t.name == "token_embd.weight" {
            println!("{}: {:?}", t.name, t.ggml_type);
        }
    }
}
