//! s-backend-cuda — backend CUDA di siliceo-motor (F3).
//!
//! Strada D1: kernels interamente in cust-rs (CUDA scritto in Rust),
//! cuBLASLt FFI solo come fallback misurato se le matmul Rust non reggono
//! il target F3 (≥40 tok/s, 9B Q4_K_M, RTX 3090).
//!
//! STADIO ATTUALE — scaffold. Nessun kernel ancora: prima si verifica
//! l'infrastruttura su hardware reale (vedi docs/F3_CUDA.md per il piano
//! completo e lo staging f16 → Q8_0/Q4_0 → K-quants). Questo modulo
//! compila senza toolchain CUDA; con `--features cuda` attiva cust.
//!
//! Regola del progetto: niente codice GPU "fidato a memoria" — ogni kernel
//! entra nel repo solo con un test di parity contro il riferimento CPU.

#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod model_qwen35;
#[cfg(feature = "cuda")]
pub mod runtime;

/// true se questo binario è stato compilato col backend CUDA.
pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(test)]
mod tests {
    #[test]
    fn feature_flag_coerente() {
        // Il flag di compilazione riflette la feature: base per il report
        // di /health quando il backend esisterà.
        assert_eq!(super::compiled_with_cuda(), cfg!(feature = "cuda"));
    }
}
