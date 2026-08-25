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

/// SMOKE TEST — scaffolding con kernel PTX scritto a mano
/// (`kernels/smoke.ptx`, somma vettori f32).
///
/// Valida l'intera catena su hardware reale: init driver → caricamento
/// modulo PTX → allocazione buffer device → copy host→device → launch →
/// copy device→host → verifica risultati.
///
/// Questo file verrà sostituito dai kernels Rust->PTX (pipeline
/// cuda_builder, decisione D1) appena la pipeline è montata; la struttura
/// di CudaCtx/launch resta quella definitiva.
#[cfg(test)]
mod smoke {
    use super::*;
    use cust::launch;
    use cust::memory::{CopyDestination, DeviceBuffer};
    use cust::module::Module;

    const N: usize = 4096;

    fn run_vec_add() -> Result<Vec<f32>, String> {
        let _ctx = CudaCtx::new(0)?;

        // cubin generato a build time da smoke.ptx (build.rs + ptxas)
        let module = Module::from_cubin(
            include_bytes!(concat!(env!("OUT_DIR"), "/smoke.cubin")) as &[u8],
            &[],
        )
        .map_err(|e| err("from_cubin", e))?;
        let func = module.get_function("vec_add").map_err(|e| err("get_function", e))?;

        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

        let d_a = DeviceBuffer::from_slice(&a).map_err(|e| err("upload a", e))?;
        let d_b = DeviceBuffer::from_slice(&b).map_err(|e| err("upload b", e))?;
        let d_out: DeviceBuffer<f32> =
            DeviceBuffer::zeroed(N).map_err(|e| err("alloc out", e))?;

        // griglia: blocchi da 256 thread, arrotondato per eccesso
        let threads = 256;
        let blocks = (N as u32).div_ceil(threads);

        use cust::stream::{Stream, StreamFlags};
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| err("stream", e))?;

        unsafe {
            launch!(
                func<<<blocks, threads, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_out.as_device_ptr(),
                    N as u32
                )
            )
            .map_err(|e| err("launch", e))?;
        }
        // il launch è asincrono: si aspetta la fine prima di copiare
        stream.synchronize().map_err(|e| err("synchronize", e))?;

        // copy_to (CopyDestination) è bloccante.
        let mut out = vec![0.0f32; N];
        d_out.copy_to(&mut out).map_err(|e| err("download", e))?;
        Ok(out)
    }

    #[test]
    fn vec_add_sulla_gpu() {
        let out = run_vec_add().expect("smoke test GPU");
        for (i, &v) in out.iter().enumerate() {
            assert!((v - N as f32).abs() < 1e-3, "out[{i}]={v}, atteso {}", N as f32);
        }
    }
}
