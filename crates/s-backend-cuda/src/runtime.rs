//! Runtime CUDA (feature `cuda`) — F3b.
//!
//! Responsabilità (in ordine di implementazione):
//! 1. ✅ `CudaCtx`: init driver, selezione device, info, errori gestiti.
//! 2. Pesi residenti su device: upload una tantum al load, zero copie
//!    host↔device nel decode (solo token ID in ingresso, logits in uscita).
//! 3. KV cache su device: stessa semantica di s-models::kv (posizioni
//!    assolute, append), buffer pre-allocati a max_seq.
//! 4. Kernels: matmul (prima f32/f16 naive poi tiled), rmsnorm, RoPE NEOX,
//!    softmax causale fused per-head, silu-mul, add residuo.
//!
//! Vincolo di correttezza: ogni kernel ha un test di parity contro la CPU
//! (stessi input → tolleranza documentata). La generazione greedy multi-token
//! end-to-end è il test definitivo, come insegnato da F1.

use cust::context::{Context, ContextFlags};
use cust::device::{Device, DeviceAttribute};
use cust::CudaFlags;

pub type DevResult = Result<(), String>;

fn err<E: std::fmt::Display>(what: &str, e: E) -> String {
    format!("cuda/{what}: {e}")
}

/// Contesto CUDA: device selezionato + contesto driver attivo.
/// RAII: al drop il contesto viene rimosso dallo stack del thread.
pub struct CudaCtx {
    device: Device,
    /// Contesto primario del device: deve restare vivo (e non drop) per
    /// tutta la durata dei buffer e dei launch creati dentro.
    #[allow(dead_code)]
    ctx: Context,
}

impl CudaCtx {
    /// Apre il device `ordinal` (contesto primario, corrente nel thread).
    pub fn new(ordinal: u32) -> Result<Self, String> {
        cust::init(CudaFlags::empty()).map_err(|e| err("init", e))?;
        let device = Device::get_device(ordinal).map_err(|e| err("get_device", e))?;
        let ctx = Context::new(device).map_err(|e| err("create_context", e))?;
        let _ = ctx.set_flags(ContextFlags::SCHED_AUTO);
        Ok(Self { device, ctx })
    }

    /// Nome, compute capability e VRAM del device — per /health e log.
    pub fn describe(&self) -> Result<String, String> {
        let name = self.device.name().map_err(|e| err("name", e))?;
        let major = self
            .device
            .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
            .map_err(|e| err("cc major", e))?;
        let minor = self
            .device
            .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
            .map_err(|e| err("cc minor", e))?;
        let mem_gb = self.device.total_memory().map_err(|e| err("total_memory", e))? as f64
            / (1024.0 * 1024.0 * 1024.0);
        Ok(format!("{name} sm_{major}{minor}, {:.1} GB", mem_gb))
    }
}

/// SMOKE + PARITY — kernel Rust compilati via cuda_builder (strada D1).
///
/// Il cubin nasce da `kernels/` (Rust → PTX via rustc_codegen_nvvm → cubin
/// via ptxas in build.rs). Qui si verifica che:
/// 1. l'intera catena driver→modulo→launch→copy-back funzioni (vec_add);
/// 2. il matmul GPU dia gli stessi risultati del riferimento CPU
///    (`matmul_rows` di s-models, stessa semantica y = x @ W^T).
#[cfg(test)]
mod smoke {
    use super::*;
    use cust::launch;
    use cust::memory::{CopyDestination, DeviceBuffer};
    use cust::module::Module;

    const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

    /// Tiene vivi contesto e modulo per tutta la durata del test:
    /// il modulo è legato al contesto — se il contesto drop, muore.
    struct Gpu {
        _ctx: CudaCtx,
        module: Module,
        stream: cust::stream::Stream,
    }

    impl Gpu {
        fn new() -> Result<Self, String> {
            let ctx = CudaCtx::new(0)?;
            let module =
                Module::from_cubin(CUBIN, &[]).map_err(|e| err("from_cubin", e))?;
            let stream = {
                use cust::stream::{Stream, StreamFlags};
                Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| err("stream", e))?
            };
            Ok(Self { _ctx: ctx, module, stream })
        }

        fn function(&self, name: &str) -> Result<cust::function::Function<'_>, String> {
            self.module.get_function(name).map_err(|e| err("get_function", e))
        }
    }

    /// Riferimento CPU: identico a matmul_rows di s-models.
    fn matmul_rows_cpu(w: &[f32], n_out: usize, n_in: usize, x: &[f32], seq: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; seq * n_out];
        for s in 0..seq {
            for o in 0..n_out {
                let w_off = o * n_in;
                let mut sum = 0.0f32;
                for d in 0..n_in {
                    sum += w[w_off + d] * x[s * n_in + d];
                }
                y[s * n_out + o] = sum;
            }
        }
        y
    }

    #[test]
    fn vec_add_rust_kernel_sulla_gpu() {
        const N: usize = 4096;
        let gpu = Gpu::new().unwrap();
        let func = gpu.function("vec_add_f32").unwrap();

        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let d_a = DeviceBuffer::from_slice(&a).unwrap();
        let d_b = DeviceBuffer::from_slice(&b).unwrap();
        let d_out = DeviceBuffer::<f32>::zeroed(N).unwrap();

        let threads = 256;
        let blocks = (N as u32).div_ceil(threads);
        // launch! accetta solo un identificatore di stream, non campi
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_out.as_device_ptr(),
                    N as u32
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        let mut out = vec![0.0f32; N];
        d_out.copy_to(&mut out).map_err(|e| err("download", e)).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - N as f32).abs() < 1e-3, "out[{i}]={v}, atteso {}", N as f32);
        }
    }

    #[test]
    fn matmul_parity_contro_cpu() {
        // dimensioni piccole ma non banali (non multiple dei blocchi)
        const SEQ: usize = 17;
        const N_OUT: usize = 33;
        const N_IN: usize = 65;
        let gpu = Gpu::new().unwrap();
        let func = gpu.function("matmul_wt_f32").unwrap();

        // input deterministici in [-1, 1): LCG semplice, niente dipendenze
        let mut seed = 0x5EED_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..SEQ * N_IN).map(|_| next()).collect();
        let w: Vec<f32> = (0..N_OUT * N_IN).map(|_| next()).collect();

        let expected = matmul_rows_cpu(&w, N_OUT, N_IN, &x, SEQ);

        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_w = DeviceBuffer::from_slice(&w).unwrap();
        let d_y = DeviceBuffer::<f32>::zeroed(SEQ * N_OUT).unwrap();

        let total = (SEQ * N_OUT) as u32;
        let threads = 256;
        let blocks = total.div_ceil(threads);
        // launch! accetta solo un identificatore di stream, non campi
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_x.as_device_ptr(),
                    d_w.as_device_ptr(),
                    d_y.as_device_ptr(),
                    SEQ as u32,
                    N_OUT as u32,
                    N_IN as u32
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        let got = {
            let mut v = vec![0.0f32; SEQ * N_OUT];
            d_y.copy_to(&mut v).map_err(|e| err("download", e)).unwrap();
            v
        };

        // stessa operazione f32 nello stesso ordine (riduzione sequenziale):
        // ci aspettiamo bit-exact; tolleranza piccola come rete di sicurezza.
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        assert!(max_err < 1e-3, "parity matmul: max_err={max_err}");
    }

    // ── riferimenti CPU: repliche fedeli delle primitive di s-models
    // (rmsnorm_into / rope_norm_offset / softmax_row / silu) ──

    fn rmsnorm_cpu(x: &[f32], w: &[f32], seq: usize, n: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; seq * n];
        for s in 0..seq {
            let row = &x[s * n..(s + 1) * n];
            let mut ss = 0.0f32;
            for &v in row {
                ss += v * v;
            }
            let inv = 1.0 / (ss / n as f32 + eps).sqrt();
            for i in 0..n {
                out[s * n + i] = row[i] * inv * w[i];
            }
        }
        out
    }

    fn rope_neox_cpu(
        x: &mut [f32],
        seq: usize,
        n_heads: usize,
        hd: usize,
        theta: f32,
        pos0: usize,
    ) {
        let half = hd / 2;
        for s in 0..seq {
            for h in 0..n_heads {
                let base = (s * n_heads + h) * hd;
                for i in 0..half {
                    let freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                    let angle = (s + pos0) as f32 * freq;
                    let (sin, cos) = angle.sin_cos();
                    let x0 = x[base + i];
                    let x1 = x[base + i + half];
                    x[base + i] = x0 * cos - x1 * sin;
                    x[base + i + half] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    fn softmax_rows_cpu(x: &mut [f32], rows: usize, n: usize) {
        for r in 0..rows {
            let row = &mut x[r * n..(r + 1) * n];
            let mut max = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    fn silu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect()
    }

    /// Input deterministico in [-1, 1): LCG semplice, niente dipendenze.
    fn lcg_input(n: usize, seed: u64) -> Vec<f32> {
        let mut seed = seed;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| next()).collect()
    }

    #[test]
    fn rmsnorm_parity_contro_cpu() {
        const SEQ: usize = 5;
        const N: usize = 96; // non multiplo del blocco
        const EPS: f32 = 1e-5; // qwen2.5 rms_eps
        let gpu = Gpu::new().unwrap();
        let func = gpu.function("rmsnorm_f32").unwrap();

        let x = lcg_input(SEQ * N, 0xA11CE);
        let w = lcg_input(N, 0xB0B);
        // pesi tipo RMSNorm attorno a 1 (come dopo training): [0.5, 1.5)
        let w: Vec<f32> = w.iter().map(|&v| 1.0 + v / 2.0).collect();
        let expected = rmsnorm_cpu(&x, &w, SEQ, N, EPS);

        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_w = DeviceBuffer::from_slice(&w).unwrap();
        let d_out = DeviceBuffer::<f32>::zeroed(SEQ * N).unwrap();

        let threads = 64;
        let blocks = (SEQ as u32).div_ceil(threads);
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_x.as_device_ptr(),
                    d_w.as_device_ptr(),
                    d_out.as_device_ptr(),
                    SEQ as u32,
                    N as u32,
                    EPS
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        let got = {
            let mut v = vec![0.0f32; SEQ * N];
            d_out.copy_to(&mut v).map_err(|e| err("download", e)).unwrap();
            v
        };
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        // stesso ordine di accumulo → atteso quasi bit-exact
        println!("rmsnorm max_err = {max_err:.3e}");
        assert!(max_err < 1e-5, "parity rmsnorm: max_err={max_err}");
    }

    #[test]
    fn rope_neox_parity_contro_cpu_con_posizione_assoluta() {
        const SEQ: usize = 3;
        const HEADS: usize = 4;
        const HD: usize = 16; // pari, half=8 coppie per testa
        const THETA: f32 = 1_000_000.0; // qwen2.5 rope_theta
        const POS0: usize = 97; // posizione assoluta ≠ 0: il caso della cache

        let gpu = Gpu::new().unwrap();
        let func = gpu.function("rope_neox_f32").unwrap();

        let mut expected = lcg_input(SEQ * HEADS * HD, 0xC0FFEE);
        let input = expected.clone();

        let d_x = DeviceBuffer::from_slice(&input).unwrap();

        let total = (SEQ * HEADS * (HD / 2)) as u32;
        let threads = 64;
        let blocks = total.div_ceil(threads);
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_x.as_device_ptr(),
                    SEQ as u32,
                    HEADS as u32,
                    HD as u32,
                    THETA,
                    POS0 as u32
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        rope_neox_cpu(&mut expected, SEQ, HEADS, HD, THETA, POS0);

        let got = {
            let mut v = vec![0.0f32; SEQ * HEADS * HD];
            d_x.copy_to(&mut v).map_err(|e| err("download", e)).unwrap();
            v
        };
        // powf/sin/cos di libdevice ≠ libm host: parity a tolleranza
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        println!("rope_neox max_err = {max_err:.3e}");
        assert!(max_err < 1e-3, "parity rope_neox: max_err={max_err}");
    }

    #[test]
    fn softmax_rows_parity_contro_cpu() {
        const ROWS: usize = 7;
        const N: usize = 33;
        let gpu = Gpu::new().unwrap();
        let func = gpu.function("softmax_rows_f32").unwrap();

        // punteggi con range realistico d'attenzione: scala * [-1,1)
        let input: Vec<f32> = lcg_input(ROWS * N, 0x50F7A5).iter().map(|&v| v * 8.0).collect();
        let mut expected = input.clone();
        softmax_rows_cpu(&mut expected, ROWS, N);

        let d_x = DeviceBuffer::from_slice(&input).unwrap();

        let threads = 32;
        let blocks = (ROWS as u32).div_ceil(threads);
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_x.as_device_ptr(),
                    ROWS as u32,
                    N as u32
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        let got = {
            let mut v = vec![0.0f32; ROWS * N];
            d_x.copy_to(&mut v).map_err(|e| err("download", e)).unwrap();
            v
        };
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        // probabilità sommano a 1: tolleranza assoluta stretta ma onesta
        // su expf di libdevice (~2 ulp)
        println!("softmax max_err = {max_err:.3e}");
        assert!(max_err < 1e-5, "parity softmax: max_err={max_err}");
    }

    #[test]
    fn silu_mul_parity_contro_cpu() {
        const N: usize = 1000;
        let gpu = Gpu::new().unwrap();
        let func = gpu.function("silu_mul_f32").unwrap();

        // range realistico post-proiezioni FFN
        let gate: Vec<f32> = lcg_input(N, 0xCAFE).iter().map(|&v| v * 5.0).collect();
        let up: Vec<f32> = lcg_input(N, 0xF00D).iter().map(|&v| v * 5.0).collect();
        let expected = silu_mul_cpu(&gate, &up);

        let d_g = DeviceBuffer::from_slice(&gate).unwrap();
        let d_u = DeviceBuffer::from_slice(&up).unwrap();
        let d_out = DeviceBuffer::<f32>::zeroed(N).unwrap();

        let threads = 256;
        let blocks = (N as u32).div_ceil(threads);
        let stream = &gpu.stream;
        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_g.as_device_ptr(),
                    d_u.as_device_ptr(),
                    d_out.as_device_ptr(),
                    N as u32
                )
            )
            .map_err(|e| err("launch", e))
            .unwrap();
        }
        gpu.stream.synchronize().map_err(|e| err("sync", e)).unwrap();

        let got = {
            let mut v = vec![0.0f32; N];
            d_out.copy_to(&mut v).map_err(|e| err("download", e)).unwrap();
            v
        };
        let mut max_err = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g - e).abs());
        }
        println!("silu_mul max_err = {max_err:.3e}");
        assert!(max_err < 1e-4, "parity silu_mul: max_err={max_err}");
    }
}
