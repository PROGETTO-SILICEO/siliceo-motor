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
use libm::{cosf, expf, powf, sinf, sqrtf};

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

/// RMSNorm per righe: out[s*n+i] = x[s*n+i] * inv * w[i],
/// con inv = 1/sqrt(mean(x_row²) + eps). Un thread per riga.
///
/// Stessa semantica di `rmsnorm_into` di s-models (stesso ordine di
/// accumulo: somma sequenziale dei quadrati, poi scala). Naivo di
/// proposito: la riduzione parallela/shared memory arriva col tuning.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn rmsnorm_f32(
    x: *const f32,
    w: *const f32,
    out: *mut f32,
    seq: u32,
    n: u32,
    eps: f32,
) {
    let s = index_1d();
    if s >= seq {
        return;
    }
    let row = x.add((s * n) as usize);
    let mut ss = 0.0f32;
    let mut i = 0u32;
    while i < n {
        let v = *row.add(i as usize);
        ss += v * v;
        i += 1;
    }
    let inv = 1.0 / sqrtf(ss / n as f32 + eps);
    while i > 0 {
        i -= 1;
        *out.add((s * n + i) as usize) = *row.add(i as usize) * inv * *w.add(i as usize);
    }
}

/// RoPE stile NEOX (half-split) con posizione assoluta di partenza,
/// in place: coppia (i, i+hd/2) della testa h del token s ruotata di
/// angle = (s + pos0) * theta^(-2i/hd).
///
/// Un thread per tripla (s, h, i<hd/2). Stessa semantica di
/// `rope_norm_offset` di s-models. sin/cos chiamati separatamente:
/// su nvptx non c'è garanzia che `sin_cos` venga lowering correttamente.
///
/// NOTA numerica: powf/sin/cos arrivano da libm → override su libdevice
/// (__nv_powf/ecc) sul device, libm host nei test CPU → parity a
/// tolleranza (~1e-4), NON bit-exact come matmul/add.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn rope_neox_f32(
    x: *mut f32,
    seq: u32,
    n_heads: u32,
    hd: u32,
    theta: f32,
    pos0: u32,
) {
    let half = hd / 2;
    let total = seq * n_heads * half; // n. coppie da ruotare
    let idx = index_1d();
    if idx >= total {
        return;
    }
    // scomposizione: idx = (s*n_heads + h)*half + i
    let i = idx % half;
    let sh = idx / half;
    let h = sh % n_heads;
    let s = sh / n_heads;

    let freq = powf(theta, -(2.0 * i as f32) / hd as f32);
    let angle = (s + pos0) as f32 * freq;
    let sin = sinf(angle);
    let cos = cosf(angle);

    let base = ((s * n_heads + h) * hd) as usize;
    let x0 = *x.add(base + i as usize);
    let x1 = *x.add(base + (i + half) as usize);
    *x.add(base + i as usize) = x0 * cos - x1 * sin;
    *x.add(base + (i + half) as usize) = x0 * sin + x1 * cos;
}

/// Softmax per righe (numerical stability con max subtraction), in place.
/// Un thread per riga, stesso ordine di `softmax_row` di s-models:
/// max scan → exp cumulativa → divisione.
///
/// Usato sui punteggi d'attenzione (una riga per query); la maschera
/// causale resta lato host (si lancia sulla porzione ..=abs_s).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn softmax_rows_f32(x: *mut f32, rows: u32, n: u32) {
    let r = index_1d();
    if r >= rows {
        return;
    }
    let row = x.add((r * n) as usize);

    let mut max = f32::NEG_INFINITY;
    let mut j = 0u32;
    while j < n {
        let v = *row.add(j as usize);
        if v > max {
            max = v;
        }
        j += 1;
    }

    let mut sum = 0.0f32;
    j = 0;
    while j < n {
        let e = expf(*row.add(j as usize) - max);
        *row.add(j as usize) = e;
        sum += e;
        j += 1;
    }
    j = 0;
    while j < n {
        *row.add(j as usize) /= sum;
        j += 1;
    }
}

/// FFN SwiGLU elementwise: out[i] = silu(gate[i]) * up[i], con
/// silu(v) = v / (1 + exp(-v)). Un thread per elemento, stessa formula
/// del forward CPU (`silu(gate[i]) * up[i]`).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn silu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: u32) {
    let i = index_1d();
    if i < n {
        let g = *gate.add(i as usize);
        let silu = g / (1.0 + expf(-g));
        *out.add(i as usize) = silu * *up.add(i as usize);
    }
}
