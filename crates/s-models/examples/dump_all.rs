use s_gguf::GgufFile;
use std::fs;
fn main() {
    let path = "/tmp/opencode/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";
    let mut f = GgufFile::open(path).unwrap();
    // dump all ssm and attention weights for all layers
    for i in 0..24 {
        let is_rec = i % 4 != 3; // full_attn_interval 4
        if is_rec {
            for name in ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_alpha.weight", "ssm_beta.weight", "ssm_norm.weight", "ssm_out.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                let tname = format!("blk.{}.{}", i, name);
                if let Ok(v) = f.tensor_data_f32(&tname) {
                    let out = format!("/tmp/weight_blk{}__{}.bin", i, name.replace('.', "_"));
                    fs::write(&out, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*4) }).unwrap();
                    // println!("dumped {} len {}", tname, v.len());
                }
            }
        } else {
            for name in ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                let tname = format!("blk.{}.{}", i, name);
                if let Ok(v) = f.tensor_data_f32(&tname) {
                    let out = format!("/tmp/weight_blk{}__{}.bin", i, name.replace('.', "_"));
                    fs::write(&out, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*4) }).unwrap();
                }
            }
        }
    }
    // token and output
    for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
        if let Ok(v) = f.tensor_data_f32(name) {
            let out = format!("/tmp/weight_{}.bin", name.replace('.', "_"));
            fs::write(&out, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*4) }).unwrap();
            println!("dumped {} len {}", name, v.len());
        }
    }
    println!("done");
}
