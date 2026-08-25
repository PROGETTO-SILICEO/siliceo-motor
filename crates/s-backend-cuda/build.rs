//! Build s-backend-cuda.
//!
//! Senza feature `cuda`: no-op (il workspace compila ovunque senza toolchain
//! NVIDIA — CI, macchine CPU, repo pubblica).
//!
//! Con feature `cuda`:
//! 1. cuda_builder compila il crate `kernels/` (Rust) a PTX — strada D1:
//!    kernel scritti in Rust, nightly pinnato in kernels/rust-toolchain.toml.
//! 2. ptxas converte il PTX in cubin — il driver dei nodi non ha JIT PTX
//!    (CUDA_ERROR_JIT_COMPILER_NOT_FOUND su la GPU di prova) e comunque evitare il JIT
//!    a runtime è la scelta giusta per un motore di inferenza.
//!
//! Requisiti sulla macchina di build: CUDA toolkit (ptxas) + toolchain
//! nightly pinnata con rust-src/rustc-dev/llvm-tools-preview. Il comando
//! cargo va lanciato con quella toolchain (es. `cargo +nightly-2026-04-02`),
//! altrimenti la compilazione del backend codegen fallisce per mismatch di
//! interni rustc. Vedi scripts/test-cuda-ai3.sh per l'ambiente completo.
//!
//! Architettura: env `SILICEO_CUDA_ARCH`, default `sm_86` (RTX 3090).
//! Per il nodo inferenza 2070: SILICEO_CUDA_ARCH=sm_75.

#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=SILICEO_CUDA_ARCH");

    // ── 1. Rust → PTX (cuda_builder, kernel in kernels/) ──
    let ptx_path = out_dir.join("kernels.ptx");
    cuda_builder::CudaBuilder::new(manifest_dir.join("kernels"))
        .copy_to(&ptx_path)
        .build()
        .expect("cuda_builder: compilazione kernel Rust a PTX fallita");

    // ── 2. PTX → cubin (ptxas, precompilato: niente JIT a runtime) ──
    let arch = std::env::var("SILICEO_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let cubin_path = out_dir.join("kernels.cubin");
    let status = std::process::Command::new("ptxas")
        .args([
            "-arch",
            &arch,
            ptx_path.to_str().unwrap(),
            "-o",
            cubin_path.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("ptxas non trovato ({e}): serve il CUDA toolkit"));
    if !status.success() {
        panic!("ptxas fallito su {} (arch={arch})", ptx_path.display());
    }
}
