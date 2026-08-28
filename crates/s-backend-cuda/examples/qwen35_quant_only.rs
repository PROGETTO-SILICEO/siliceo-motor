use s_models::qwen35::Model as CpuModel;
use s_models::Device;
use s_backend_cuda::model_qwen35::GpuQwen35;
fn main() -> Result<(), String> {
    let q4_path = "/home/alforiva/inference/models/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";
    eprintln!("load CPU q4 {}", q4_path);
    let cpu = CpuModel::load(q4_path).map_err(|e| e.to_string())?;
    eprintln!("CPU loaded");
    let tokens = vec![9419u32, 11, 4858, 0];
    let cpu_logits = cpu.forward(&tokens);
    eprintln!("CPU forward done");
    let mut idx: Vec<usize> = (0..cpu_logits.len()).collect();
    idx.sort_by(|a,b| cpu_logits[*b].partial_cmp(&cpu_logits[*a]).unwrap());
    for i in 0..3 { eprintln!("CPU {} {} {:.4}", i, idx[i], cpu_logits[idx[i]]); }
    eprintln!("load GPU quantized");
    let gpu = GpuQwen35::load_quantized(q4_path, 64).map_err(|e| e.to_string())?;
    eprintln!("GPU {}", gpu.describe()?);
    let mut cache = gpu.new_cache(64);
    let gpu_logits = gpu.forward_cached(&tokens, &mut cache);
    eprintln!("GPU forward done");
    let mut maxdiff = 0f32;
    for i in 0..cpu_logits.len() { maxdiff = maxdiff.max((cpu_logits[i]-gpu_logits[i]).abs()); }
    eprintln!("Q4 maxdiff {:.6e}", maxdiff);
    Ok(())
}
