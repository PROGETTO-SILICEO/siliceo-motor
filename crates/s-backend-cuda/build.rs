//! Build s-backend-cuda.
//!
//! Con la feature `cuda`: compila i kernel PTX (`src/kernels/*.ptx`) in
//! cubin con `ptxas` — il driver di alcuni nodi non ha il JIT PTX
//! (CUDA_ERROR_JIT_COMPILER_NOT_FOUND) e comunque evitare il JIT a runtime
//! è la scelta giusta per un motore di inferenza (avvio deterministico).
//!
//! Architettura target: env `SILICEO_CUDA_ARCH`, default `sm_86` (RTX 3090,
//! GPU di prova). Per il 2070 (sm_75): SILICEO_CUDA_ARCH=sm_75.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if !cfg!(feature = "cuda") {
        return;
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let arch =
        env::var("SILICEO_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    println!("cargo:rerun-if-env-changed=SILICEO_CUDA_ARCH");
    println!("cargo:rerun-if-changed=src/kernels/");

    // elenco statico dei kernel (build.rs non può globare in modo stabile)
    let kernels = ["smoke"];
    for name in kernels {
        let ptx = format!("src/kernels/{name}.ptx");
        println!("cargo:rerun-if-changed={ptx}");
        let cubin = out_dir.join(format!("{name}.cubin"));
        let status = Command::new("ptxas")
            .args(["-arch", &arch, &ptx, "-o", cubin.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| panic!("ptxas non trovato ({e}): serve il CUDA toolkit"));
        if !status.success() {
            panic!("ptxas fallito su {ptx} (arch={arch})");
        }
    }
}
