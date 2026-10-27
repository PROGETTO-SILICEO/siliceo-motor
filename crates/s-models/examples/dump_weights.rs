use s_gguf::GgufFile;
fn main() {
    let path = "/tmp/opencode/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";
    let mut f = GgufFile::open(path).unwrap();
    for name in ["blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight", "blk.3.attn_output.weight",
    "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight", "blk.3.attn_norm.weight", "blk.3.post_attention_norm.weight",
    "blk.3.ffn_gate.weight", "blk.3.ffn_up.weight", "blk.3.ffn_down.weight"] {
        match f.tensor_data_f32(name) {
            Ok(v) => {
                // save to file
                let out = format!("/tmp/weight_{}.bin", name.replace('.', "_"));
                std::fs::write(&out, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*4) }).unwrap();
                println!("{} len {} first3 {:?}", name, v.len(), &v[..3.min(v.len())]);
            },
            Err(e) => println!("{} err {}", name, e),
        }
    }
}
