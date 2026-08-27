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

#![allow(unsafe_op_in_unsafe_fn)]
// intrinsics nvptx ancora unstable sul nightly pinnato
#![feature(stdarch_nvptx)]

use cuda_std_macros::kernel;
use libm::{cosf, expf, logf, powf, sinf, sqrtf};

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

/// Bias broadcast per righe: x[s*n+i] += bias[i]. Un thread per elemento.
/// has_bias = 0 → nessun bias (modelli senza): il kernel esce subito.
/// NOTA: flag esplicito invece di puntatore NULL — il launch di cust impone
/// DeviceCopy sui parametri e i nullable attraverso la macro sono fragili;
/// il flag è anche più leggibile nel PTX generato.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn add_bias_f32(x: *mut f32, bias: *const f32, seq: u32, n: u32, has_bias: u32) {
    if has_bias == 0 {
        return;
    }
    let idx = index_1d();
    if idx >= seq * n {
        return;
    }
    let i = idx % n;
    *x.add(idx as usize) += *bias.add(i as usize);
}

/// Embedding lookup: out[s*ne+i] = table[ids[s]*ne+i]. Un thread per elemento.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gather_rows_f32(
    table: *const f32,
    ids: *const u32,
    out: *mut f32,
    seq: u32,
    ne: u32,
) {
    let idx = index_1d();
    if idx >= seq * ne {
        return;
    }
    let s = idx / ne;
    let i = idx % ne;
    let tok = *ids.add(s as usize) as usize;
    *out.add(idx as usize) = *table.add(tok * ne as usize + i as usize);
}

/// Copia n f32: dst[0..n] = src[0..n]. Serve per l'append K/V nella cache
/// device a offset arbitrario (dst arriva già offsettato dal launch).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn copy_n_f32(src: *const f32, dst: *mut f32, n: u32) {
    let i = index_1d();
    if i < n {
        *dst.add(i as usize) = *src.add(i as usize);
    }
}

/// Punteggi d'attenzione causali: un thread per (s, h).
///
/// scores[(s*n_heads+h)*W + t] = dot(q[s,h], k[t,kvh]) * scale per
/// t <= pos0+s, -INF per t > pos0+s fino a W (= pos0+seq, larghezza
/// uniforme delle righe). UNA RIGA PER TESTA: le teste NON condividono la
/// riga s (bug trovato dal parity test — le si sovrascrivevano a vicenda).
///
/// Il padding -INF permette di riusare softmax_rows_f32 su righe larghe W:
/// exp(-INF - max) = 0 esatto, e aggiungere 0.0 non altera la somma bitwise.
///
/// L'ordine di accumulo del dot (d sequenziale) è IDENTICO al forward CPU:
/// è la condizione per la parity stretta.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn attention_scores_f32(
    q: *const f32,        // [seq × n_heads*hd]
    k_cache: *const f32,  // [n_layers][max_seq][kv_dim], qui offsettato al layer
    scores: *mut f32,     // [seq × n_heads × W]
    seq: u32,
    pos0: u32,
    w_width: u32,         // W = pos0 + seq
    n_heads: u32,
    n_kv_heads: u32,
    hd: u32,
    kv_dim: u32,
    scale: f32,
) {
    let idx = index_1d(); // s * n_heads + h
    if idx >= seq * n_heads {
        return;
    }
    let s = idx / n_heads;
    let h = idx % n_heads;
    let heads_per_kv = n_heads / n_kv_heads;
    let kvh = h / heads_per_kv;

    let q_row = q.add((s * n_heads * hd) as usize).add((h * hd) as usize);
    let abs_s = pos0 + s;

    let sc_row = scores.add((idx * w_width) as usize);
    let mut t = 0u32;
    while t < w_width {
        if t <= abs_s {
            let k_row = k_cache.add((t * kv_dim) as usize).add((kvh * hd) as usize);
            let mut dot = 0.0f32;
            let mut d = 0u32;
            while d < hd {
                dot += *q_row.add(d as usize) * *k_row.add(d as usize);
                d += 1;
            }
            *sc_row.add(t as usize) = dot * scale;
        } else {
            *sc_row.add(t as usize) = f32::NEG_INFINITY;
        }
        t += 1;
    }
}

/// Somma pesata dei V: un thread per elemento di output (s, h, d).
///
/// out[s,h,d] = Σ_t weights[(s*n_heads+h)*W + t] * v_cache[t,kvh,d],
/// con t <= pos0+s (bound causale esplicito: non ci affidiamo al padding).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn attn_weighted_sum_f32(
    weights: *const f32,  // [seq × n_heads × W] già softmaxate
    v_cache: *const f32,  // offsettato al layer
    out: *mut f32,        // [seq × n_heads*hd]
    seq: u32,
    pos0: u32,
    w_width: u32,
    n_heads: u32,
    n_kv_heads: u32,
    hd: u32,
    kv_dim: u32,
) {
    let idx = index_1d(); // s * n_heads*hd + h*hd + d
    let total = seq * n_heads * hd;
    if idx >= total {
        return;
    }
    let s = idx / (n_heads * hd);
    let rem = idx % (n_heads * hd);
    let h = rem / hd;
    let d = rem % hd;
    let heads_per_kv = n_heads / n_kv_heads;
    let kvh = h / heads_per_kv;

    let w_row = weights.add(((s * n_heads + h) * w_width) as usize);
    let o = out.add(idx as usize);

    let abs_s = pos0 + s;
    let mut acc = 0.0f32;
    let mut t = 0u32;
    while t <= abs_s {
        let v = v_cache.add((t * kv_dim) as usize).add((kvh * hd) as usize).add(d as usize);
        acc += *w_row.add(t as usize) * *v;
        t += 1;
    }
    *o = acc;
}

/// SiLU in place: x[i] = x[i] * sigmoid(x[i]) = x / (1+exp(-x)).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn silu_inplace_f32(x: *mut f32, n: u32) {
    let i = index_1d();
    if i < n {
        let v = *x.add(i as usize);
        *x.add(i as usize) = v / (1.0 + expf(-v));
    }
}

/// L2 norm per testa: per ogni testa h, x[h*d_state..(h+1)*d_state] /= sqrt(sum(x²)+eps).
/// eps = 1e-6 come HF/FLA.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn l2norm_per_head_f32(x: *mut f32, n_heads: u32, d_state: u32) {
    let h = index_1d();
    if h >= n_heads {
        return;
    }
    let base = (h * d_state) as usize;
    let mut ss = 0.0f32;
    let mut i = 0u32;
    while i < d_state {
        let v = *x.add(base + i as usize);
        ss += v * v;
        i += 1;
    }
    let inv = 1.0 / sqrtf(ss + 1e-6);
    i = 0;
    while i < d_state {
        *x.add(base + i as usize) *= inv;
        i += 1;
    }
}

/// Conv1D depthwise causale: conv_buf[K][D] shift + append + conv.
/// conv_buf layout [K][D] row-major (K=conv_k, D=conv_dim).
/// Weight layout [D][K] (d*K + k) come GGUF/HF.
/// qkv_mixed = nuovo token [D], conv_out = output [D].
/// Esegue: shift rows [1..K-1] -> [0..K-2], write qkv_mixed in row K-1, poi conv.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn conv1d_depthwise_f32(
    conv_buf: *mut f32,
    qkv_mixed: *const f32,
    conv_weight: *const f32,
    conv_out: *mut f32,
    conv_k: u32,
    conv_dim: u32,
) {
    let d = index_1d();
    if d >= conv_dim {
        return;
    }
    let row = conv_dim as usize;
    // shift: per d fissato, muovi k+1 -> k
    let mut k = 0u32;
    while k + 1 < conv_k {
        *conv_buf.add((k as usize * row) + d as usize) =
            *conv_buf.add(((k + 1) as usize * row) + d as usize);
        k += 1;
    }
    // append nuovo
    *conv_buf.add(((conv_k - 1) as usize * row) + d as usize) = *qkv_mixed.add(d as usize);
    // conv: sum_k w[d*K + k] * buf[k][d]
    let mut sum = 0.0f32;
    k = 0;
    while k < conv_k {
        let w = *conv_weight.add((d * conv_k + k) as usize);
        let b = *conv_buf.add((k as usize * row) + d as usize);
        sum += w * b;
        k += 1;
    }
    *conv_out.add(d as usize) = sum;
}

/// Gate per DeltaNet: g[h] = -exp(A_log[h]) * softplus(alpha_dot[h] + dt_bias[h]).
/// alpha_dot = alpha^T @ normed (già matmul), qui solo elementwise.
/// Un thread per testa.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn delta_gate_f32(
    alpha_dot: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    g_out: *mut f32,
    n_heads: u32,
) {
    let h = index_1d();
    if h >= n_heads {
        return;
    }
    let a = *alpha_dot.add(h as usize) + *dt_bias.add(h as usize);
    let sp = if a > 20.0 { a } else { logf(1.0 + expf(a)) };
    let a_log = *ssm_a.add(h as usize);
    *g_out.add(h as usize) = -expf(a_log) * sp;
}

/// Beta per DeltaNet: beta[h] = sigmoid(beta_dot[h]).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn delta_beta_f32(beta_dot: *const f32, beta_out: *mut f32, n_heads: u32) {
    let h = index_1d();
    if h >= n_heads {
        return;
    }
    let v = *beta_dot.add(h as usize);
    *beta_out.add(h as usize) = 1.0 / (1.0 + expf(-v));
}

/// DeltaNet recurrence fused per token (single token, seq=1).
/// State layout per testa: [d_state × d_state] row-major, n_heads teste.
/// Q/K/V per testa: [d_state] (Q/K) e [head_dim] (V, qui == d_state).
/// g/beta per testa: scalar.
/// out: [n_heads * head_dim] output.
///
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn delta_recurrent_step_f32(
    state: *mut f32,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    out: *mut f32,
    n_heads: u32,
    d_state: u32,
    head_dim: u32,
) {
    let h = index_1d();
    if h >= n_heads {
        return;
    }
    let d = d_state as usize;
    let hd = head_dim as usize;
    let base_state = (h as usize) * d * d;
    let base_qkv = (h as usize) * d;
    let base_v = (h as usize) * hd;

    let q_h = q.add(base_qkv);
    let k_h = k.add(base_qkv);
    let v_h = v.add(base_v);
    let g_val = *g.add(h as usize);
    let b_val = *beta.add(h as usize);
    let g_exp = expf(g_val);

    // decay: S *= exp(g)
    let mut idx = 0usize;
    while idx < d * d {
        *state.add(base_state + idx) *= g_exp;
        idx += 1;
    }

    // sk = S @ k  [d]
    let mut sk = [0.0f32; 128];
    let mut j = 0u32;
    while j < d_state {
        let mut sum = 0.0f32;
        let mut i = 0u32;
        while i < d_state {
            sum += *state.add(base_state + (j as usize * d + i as usize)) * *k_h.add(i as usize);
            i += 1;
        }
        sk[j as usize] = sum;
        j += 1;
    }

    // S += k ⊗ ((v - sk)*beta)  — head_dim == d_state per qwen35-0.8B
    j = 0;
    while j < d_state {
        let d_val = (*v_h.add(j as usize) - sk[j as usize]) * b_val;
        let mut i = 0u32;
        while i < d_state {
            *state.add(base_state + (j as usize * d + i as usize)) += *k_h.add(i as usize) * d_val;
            i += 1;
        }
        j += 1;
    }

    // o = S @ q
    let out_h = out.add(base_v);
    j = 0;
    while j < head_dim {
        let mut sum = 0.0f32;
        let mut i = 0u32;
        while i < d_state {
            sum += *state.add(base_state + (j as usize * d + i as usize)) * *q_h.add(i as usize);
            i += 1;
        }
        *out_h.add(j as usize) = sum;
        j += 1;
    }
}

/// RoPE MRoPE parziale: ruota solo i primi n_rot dims per testa (half = n_rot/2).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn rope_mrope_f32(
    x: *mut f32,
    seq: u32,
    n_heads: u32,
    hd: u32,
    theta: f32,
    pos0: u32,
    n_rot: u32,
) {
    let half = hd / 2;
    let half_rot = n_rot / 2;
    let total = seq * n_heads * half_rot;
    let idx = index_1d();
    if idx >= total {
        return;
    }
    let i = idx % half_rot;
    let sh = idx / half_rot;
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

/// RMSNorm per testa: per ogni testa h, x[h*hd..(h+1)*hd] /= sqrt(mean(x²)+eps) * w[h*hd..].
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn rmsnorm_per_head_f32(
    x: *mut f32,
    w: *const f32,
    n_heads: u32,
    hd: u32,
    eps: f32,
) {
    let h = index_1d();
    if h >= n_heads {
        return;
    }
    let base = (h * hd) as usize;
    let mut ss = 0.0f32;
    let mut i = 0u32;
    while i < hd {
        let v = *x.add(base + i as usize);
        ss += v * v;
        i += 1;
    }
    let inv = 1.0 / sqrtf(ss / hd as f32 + eps);
    i = 0;
    while i < hd {
        *x.add(base + i as usize) = *x.add(base + i as usize) * inv * *w.add(base + i as usize);
        i += 1;
    }
}
