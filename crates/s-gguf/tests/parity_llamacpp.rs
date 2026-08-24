//! Parity test F0: il nostro parser contro llama.cpp `llama-gguf` su file reale.
//!
//! Riferimento misurato con `llama-gguf file r n`:
//!   version 3, alignment 32, data offset 5947744, n_kv 26, n_tensors 291
//!   tensor[0] = output.weight  size=144643072 offset=0
//!   tensor[1] = token_embd.weight size=93592576 offset=144643072
//!   tensor[2] = blk.0.attn_norm.weight size=3584 offset=238235648
//!
//! Attivazione: S_GGUF_QWEN05B=<percorso file>  cargo test -p s-gguf --test parity_llamacpp

#[test]
fn parity_with_llama_gguf() {
    let Ok(path) = std::env::var("S_GGUF_QWEN05B") else { return; };

    // --- riferimento (misurato con llama-gguf di llama.cpp) ---
    const REF_N_TENSORS: u64 = 291;
    const REF_N_KV: u64 = 26;
    const REF_ALIGNMENT: u64 = 32;
    const REF_DATA_OFFSET: u64 = 5_947_744;

    let f = s_gguf::GgufFile::open(&path).unwrap_or_else(|e| panic!("apertura fallita: {e}"));

    assert_eq!(f.header.version, 3, "version");
    assert_eq!(f.header.kv_count, REF_N_KV, "kv_count");
    assert_eq!(f.header.tensor_count, REF_N_TENSORS, "tensor_count");
    assert_eq!(f.alignment(), REF_ALIGNMENT, "alignment");
    assert_eq!(f.data_start(), REF_DATA_OFFSET, "data offset");

    // tensor[0]: output.weight — Q6_K su qwen2.5 (size 144643072 = 151936*256/256*210? no:
    // verificato empiricamente dal dump: size in byte 144643072)
    let t0 = f.tensor("output.weight").unwrap();
    assert_eq!(t0.offset, 0, "output.weight offset");
    assert_eq!(t0.n_bytes(), 144_643_072, "output.weight size");

    // tensor[1]: token_embd.weight
    let t1 = f.tensor("token_embd.weight").unwrap();
    assert_eq!(t1.offset, 144_643_072, "token_embd offset");
    assert_eq!(t1.n_bytes(), 93_592_576, "token_embd size");

    // tensor[2]: blk.0.attn_norm (F32)
    let t2 = f.tensor("blk.0.attn_norm.weight").unwrap();
    assert_eq!(t2.offset, 238_235_648, "attn_norm offset");
    assert_eq!(t2.n_bytes(), 3584, "attn_norm size");

    // Ordine dei tensori identico al dump (i primi 8 nomi)
    let expected_order = [
        "output.weight",
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_k.bias",
    ];
    for (i, name) in expected_order.iter().enumerate() {
        assert_eq!(&f.tensors[i].name, name, "ordine tensori posizione {i}");
    }

    // Metadata chiave presenti
    for k in ["general.architecture", "qwen2.block_count", "tokenizer.ggml.tokens"] {
        assert!(f.metadata.iter().any(|(key, _)| key == k), "manca KV {k}");
    }

    println!("✅ PARITY F0: 291 tensori, offset e dimensioni identici a llama-gguf");
}
