use s_models::qwen35::Model as CpuModel;
use s_models::Device;
use s_backend_cuda::model_qwen35::GpuQwen35;
fn run_f16() -> Result<bool, String> {
    let f16_path = "/tmp/Qwen3.5-0.8B-F16.gguf";
    eprintln!("=== F16 ===");
    let cpu = CpuModel::load(f16_path).map_err(|e| e.to_string())?;
    let gpu = GpuQwen35::load(&cpu, 16).map_err(|e| e.to_string())?;
    let tokens = vec![9419u32, 11, 4858, 0];
    let cpu_logits = cpu.forward(&tokens);
    let mut cache = gpu.new_cache(16);
    let gpu_logits = gpu.forward_cached(&tokens, &mut cache);
    let mut maxdiff = 0f32;
    for i in 0..cpu_logits.len() { maxdiff = maxdiff.max((cpu_logits[i]-gpu_logits[i]).abs()); }
    eprintln!("F16 maxdiff {:.6e}", maxdiff);
    if maxdiff < 1e-4 { eprintln!("F16 ✅"); } else { eprintln!("F16 FAIL"); }
    Ok(maxdiff < 1e-4)
}

fn run_q4() -> Result<(), String> {
    let q4_path = "/home/alforiva/inference/models/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";
    eprintln!("=== Q4_K_M quantized ===");
    let cpu = CpuModel::load(q4_path).map_err(|e| e.to_string())?;
    let gpu = GpuQwen35::load_quantized(q4_path, 64).map_err(|e| e.to_string())?;
    eprintln!("GPU {}", gpu.describe()?);
    let tokens = vec![9419u32, 11, 4858, 0];
    let cpu_logits = cpu.forward(&tokens);
    let mut cache = gpu.new_cache(64);
    let gpu_logits = gpu.forward_cached(&tokens, &mut cache);
    let mut maxdiff = 0f32; let mut mean = 0f32;
    for i in 0..cpu_logits.len() { let d=(cpu_logits[i]-gpu_logits[i]).abs(); maxdiff=maxdiff.max(d); mean+=d; }
    mean/=cpu_logits.len() as f32;
    eprintln!("Q4 maxdiff {:.6e} mean {:.6e}", maxdiff, mean);
    let mut idx_cpu: Vec<usize> = (0..cpu_logits.len()).collect();
    idx_cpu.sort_by(|a,b| cpu_logits[*b].partial_cmp(&cpu_logits[*a]).unwrap());
    let mut idx_gpu: Vec<usize> = (0..gpu_logits.len()).collect();
    idx_gpu.sort_by(|a,b| gpu_logits[*b].partial_cmp(&gpu_logits[*a]).unwrap());
    for i in 0..5 { eprintln!("CPU {} {} {:.4} | GPU {} {:.4}", i, idx_cpu[i], cpu_logits[idx_cpu[i]], idx_gpu[i], gpu_logits[idx_gpu[i]]); }
    eprintln!("top5 identical? {}", idx_cpu[..5]==idx_gpu[..5]);
    if maxdiff < 5e-3 { eprintln!("Q4 ✅ parity OK"); } else { eprintln!("Q4 ⚠️ FAIL"); }
    let mut cache2 = gpu.new_cache(64);
    let _ = gpu.forward_cached(&tokens[..1], &mut cache2);
    let inc = gpu.forward_cached(&tokens[1..], &mut cache2);
    let full = { let mut c = gpu.new_cache(64); gpu.forward_cached(&tokens, &mut c) };
    let mut maxdiff_inc = 0f32;
    for i in 0..full.len() { maxdiff_inc=maxdiff_inc.max((full[i]-inc[i]).abs()); }
    eprintln!("incremental vs full maxdiff {:.6e}", maxdiff_inc);
    Ok(())
}

fn main() -> Result<(), String> {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "--f16-only" => { run_f16()?; Ok(()) },
        "--q4-only" => run_q4(),
        _ => {
            // Combined: run in subprocesses to avoid stacking 2 CUDA contexts in same process (cust segfault)
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            eprintln!("→ subprocess F16");
            let st = std::process::Command::new(&exe).arg("--f16-only").status().map_err(|e| e.to_string())?;
            if !st.success() { return Err(format!("F16 subprocess failed: {:?}", st)); }
            eprintln!("→ subprocess Q4");
            let st = std::process::Command::new(&exe).arg("--q4-only").status().map_err(|e| e.to_string())?;
            if !st.success() { return Err(format!("Q4 subprocess failed: {:?}", st)); }
            eprintln!("✅ both subprocesses passed");
            Ok(())
        }
    }
}
