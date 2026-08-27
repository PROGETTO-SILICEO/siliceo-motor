use s_models::qwen35::Model;
use std::fs;
fn main() {
    let path = "/tmp/opencode/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";
    let model = Model::load(path).unwrap();
    for li in [3usize,7,11,15] {
        let hs_in = fs::read(format!("/tmp/hs_{}.bin", li)).unwrap();
        let hs_in_f32: Vec<f32> = hs_in.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let out = model.debug_single_layer(li, &hs_in_f32);
        let hs_out_expected = fs::read(format!("/tmp/hs_{}.bin", li+1)).unwrap();
        let hs_out_exp: Vec<f32> = hs_out_expected.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let diff: f32 = out.iter().zip(hs_out_exp.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / out.len() as f32;
        let maxdiff = out.iter().zip(hs_out_exp.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        println!("layer {} in first3 {:?} -> out first3 {:?} expected {:?} avgdiff {:.4} maxdiff {:.4} norm out {:.2} exp {:.2}", li, &hs_in_f32[..3], &out[..3], &hs_out_exp[..3], diff, maxdiff, out.iter().map(|v| v*v).sum::<f32>().sqrt(), hs_out_exp.iter().map(|v| v*v).sum::<f32>().sqrt());
        // also check full vector diff for first attention layer detail
        if li==3 {
            // dump first 10 diff
            for i in 0..10 {
                println!("  dim {} rust {:.4} hf {:.4} diff {:.4}", i, out[i], hs_out_exp[i], out[i]-hs_out_exp[i]);
            }
        }
    }
}
