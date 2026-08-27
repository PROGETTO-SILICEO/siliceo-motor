use s_models::qwen35::Model as CpuModel;
use s_models::Device;
use s_backend_cuda::model_qwen35::GpuQwen35;
fn main() -> Result<(), String> {
    let path = "/tmp/Qwen3.5-0.8B-F16.gguf";
    eprintln!("load CPU {}", path);
    let cpu = CpuModel::load(path).map_err(|e| e.to_string())?;
    eprintln!("CPU config: {} layers, embd {}", cpu.config.n_layers, cpu.config.n_embd);
    let tokens = vec![9419u32];
    let cpu_logits = cpu.forward(&tokens);
    let mut idx: Vec<usize> = (0..cpu_logits.len()).collect();
    idx.sort_by(|a,b| cpu_logits[*b].partial_cmp(&cpu_logits[*a]).unwrap());
    for i in 0..5 { eprintln!("CPU {} {} {:.4}", i, idx[i], cpu_logits[idx[i]]); }
    eprintln!("load GPU max_seq 16");
    let gpu = GpuQwen35::load(&cpu, 16).map_err(|e| e.to_string())?;
    eprintln!("GPU {}", gpu.describe()?);
    let mut cache = gpu.new_cache(16);
    let gpu_logits = gpu.forward_cached(&tokens, &mut cache);
    let mut idx2: Vec<usize> = (0..gpu_logits.len()).collect();
    idx2.sort_by(|a,b| gpu_logits[*b].partial_cmp(&gpu_logits[*a]).unwrap());
    for i in 0..5 { eprintln!("GPU {} {} {:.4}", i, idx2[i], gpu_logits[idx2[i]]); }
    let mut maxdiff = 0f32;
    let mut meandiff = 0f32;
    for i in 0..cpu_logits.len() { let d=(cpu_logits[i]-gpu_logits[i]).abs(); maxdiff=maxdiff.max(d); meandiff+=d; }
    meandiff/=cpu_logits.len() as f32;
    eprintln!("maxdiff {:.6} mean {:.6}", maxdiff, meandiff);
    eprintln!("top5 identical? {}", idx[..5]==idx2[..5]);
    if maxdiff < 0.01 { eprintln!("✅ parity OK"); } else { eprintln!("⚠️ parity FAIL >0.01"); }
    let seq2 = vec![9419u32, 11u32, 0u32];
    let cpu_logits2 = cpu.forward(&seq2);
    let mut cpu_cache = cpu.new_cache(16);
    let _ = cpu.forward_cached(&seq2[..1], &mut cpu_cache);
    let cpu_cached = cpu.forward_cached(&seq2[1..], &mut cpu_cache);
    eprintln!("CPU cached vs full maxdiff: {:.6}", cpu_logits2.iter().zip(cpu_cached.iter()).map(|(a,b)|(a-b).abs()).fold(0f32, f32::max));
    let mut gpu_cache2 = gpu.new_cache(16);
    let _ = gpu.forward_cached(&seq2[..1], &mut gpu_cache2);
    let gpu_cached = gpu.forward_cached(&seq2[1..], &mut gpu_cache2);
    let maxd2 = cpu_logits2.iter().zip(gpu_cached.iter()).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
    eprintln!("GPU incremental vs CPU full maxdiff {:.6}", maxd2);
    if maxd2 < 0.01 { eprintln!("✅ incremental parity OK"); } else { eprintln!("⚠️ incremental FAIL"); }
    Ok(())
}
