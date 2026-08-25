//! Kernel GPU di siliceo-motor — Rust compilato a PTX (strada D1).
//!
//! Compilato da cuda_builder con target nvptx64-nvidia-cuda; il PTX risultante
//! viene convertito in cubin da ptxas nel build.rs di s-backend-cuda
//! (il driver dei nodi non ha JIT PTX: tutto precompilato, avvio deterministico).
//!
//! Dipendenze MINIME di proposito: solo la proc-macro #[kernel] di
//! cuda_std_macros. Gli indici thread/block arrivano dagli intrinsics nativi
//! di rustc per nvptx (core::arch::nvptx) — NON da cuda_std, che tira dentro
//  half/glam/vek e oggi non compila su nvptx (half abilita `std` di default).
//!
//! Convenzioni:
//! - firme con puntatori espliciti (niente slice): mappatura 1:1 con i
//!   parametri del launch di cust;
//! - ogni kernel ha un test di parity contro la CPU in s-backend-cuda;
//! - niente ottimizzazioni finché la parity non è verde.

// intrinsics nvptx ancora unstable sul nightly pinnato
#![feature(stdarch_nvptx)]

use cuda_std_macros::kernel;

/// Indice globale 1-D del thread (grid monodimensionale).
#[inline(always)]
fn index_1d() -> u32 {
    // intrinsics nativi del target nvptx64 (nightly)
    unsafe { core::arch::nvptx::_block_idx_x() * core::arch::nvptx::_block_dim_x() + core::arch::nvptx::_thread_idx_x() }
}

/// out[i] = a[i] + b[i] — smoke/parity della pipeline Rust→PTX→cubin.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn vec_add_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let i = index_1d();
    if i < n {
        *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
    }
}

/// y[seq×n_out] = x[seq×n_in] @ W[n_out×n_in]^T  (row-major, W per righe).
///
/// Stessa semantica di `matmul_rows` CPU: un thread per elemento di output,
/// riduzione interna su n_in. Naivo di proposito: prima la parity, poi il tuning.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn matmul_wt_f32(
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    seq: u32,
    n_out: u32,
    n_in: u32,
) {
    let idx = index_1d(); // s * n_out + o
    if idx >= seq * n_out {
        return;
    }
    let s = idx / n_out;
    let o = idx % n_out;
    // righe flat: la riga s di x inizia a s*n_in, la riga o di W a o*n_in
    let x_row = x.add((s * n_in) as usize);
    let w_row = w.add((o * n_in) as usize);

    let mut acc = 0.0f32;
    let mut d = 0u32;
    while d < n_in {
        acc += *x_row.add(d as usize) * *w_row.add(d as usize);
        d += 1;
    }
    *y.add(idx as usize) = acc;
}
