//! Dequantizzazione GGML → f32.
//!
//! Gli algoritmi Q4_K, Q8_0, Q6_K sono PORTATI da Exo v0.17-0.20
//! (src/tensor.rs), dove sono stati verificati BIT-PERFECT contro l'implementazione
//! Python di riferimento sul file reale LFM2.5 (std=0.0152). Non modificarli
//! senza rieseguire i test di parity.

use crate::{GgufError, Result, TensorInfo};
use crate::types::GgmlType;

/// fp16 (bit pattern) → f32. Implementazione manuale verificata (Exo).
pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    let (e, f): (u32, u32) = if exp == 0 {
        if frac == 0 {
            (0, 0)
        } else {
            // subnormale fp16: valore = frac * 2^-24, esatto in f32
            let val = (frac as f32) * 5.960_464_5e-8;
            return if sign == 1 { -val } else { val };
        }
    } else if exp == 0x1F {
        (0xFF, if frac == 0 { 0 } else { 0x200_000 })
    } else {
        (exp + 127 - 15, frac)
    };

    let bits = (sign << 31) | (e << 23) | (f << 13);
    f32::from_bits(bits)
}

/// bf16 (bit pattern) → f32: basta un shift (bf16 è la metà alta di f32).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// get_scale_min_k4 da ggml-quants.c (pattern scala/min dei sub-block Q4_K).
pub fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantizza un blocco Q8_0 (34 byte → 32 f32). Layout: [d:fp16][qs:32×i8].
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let d = fp16_to_f32(block[0] as u16 | ((block[1] as u16) << 8));
    for i in 0..32 {
        out[i] = (block[2 + i] as i8) as f32 * d;
    }
}

/// Dequantizza un blocco Q5_0 (22 byte → 32 f32).
/// Layout: [d:fp16][qh:4 byte][qs:16 byte nibbles].
/// Formula (verificata contro gguf-py ufficiale + pesi HF Qwen2.5, corr 0.9992):
/// q_v = (nibble_v | bit_v<<4) − 16, dove bit_v è il bit v dell'u32
/// little-endian formato dai 4 byte qh. Nibble: low di qs[v] per v<16,
/// high di qs[v−16] per v≥16. y = d*q.
pub fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    let d = fp16_to_f32(block[0] as u16 | ((block[1] as u16) << 8));
    let qh_u32 = block[2] as u32
        | ((block[3] as u32) << 8)
        | ((block[4] as u32) << 16)
        | ((block[5] as u32) << 24);
    let qs = &block[6..22];
    for l in 0..16 {
        let hi_lo = (((qh_u32 >> l) & 1) << 4) as i32;
        let hi_hi = (((qh_u32 >> (l + 16)) & 1) << 4) as i32;
        let q_lo = ((qs[l] & 0x0F) as i32 | hi_lo) - 16;
        let q_hi = ((qs[l] >> 4) as i32 | hi_hi) - 16;
        out[l] = d * q_lo as f32;
        out[l + 16] = d * q_hi as f32;
    }
}

/// Dequantizza un blocco Q4_K (144 byte → 256 f32). Layout: [d:fp16][dmin:fp16]
/// [scales:12][qs:128]. Formula dequantize_row_q4_K: x = d*ds*q - dmin*ms.
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    let d = fp16_to_f32(block[0] as u16 | ((block[1] as u16) << 8));
    let mn = fp16_to_f32(block[2] as u16 | ((block[3] as u16) << 8));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut is = 0usize;
    let mut idx = 0usize;
    let mut qoff = 0usize;
    for _j in 0..(256 / 64) {
        let (ds1, ms1) = get_scale_min_k4(is, scales);
        let d1 = d * ds1 as f32;
        let m1 = mn * ms1 as f32;
        let (ds2, ms2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * ds2 as f32;
        let m2 = mn * ms2 as f32;
        for l in 0..32 {
            out[idx + l] = d1 * (qs[qoff + l] & 0x0F) as f32 - m1;
        }
        for l in 0..32 {
            out[idx + 32 + l] = d2 * (qs[qoff + l] >> 4) as f32 - m2;
        }
        qoff += 32;
        idx += 64;
        is += 2;
    }
}

/// Dequantizza un blocco Q6_K (210 byte → 256 f32).
/// Layout UFFICIALE (gguf-py/ggml, verificato su file reale):
/// [ql:128][qh:64][scales:16×i8][d:fp16] — la scala d sta ALLA FINE del blocco.
/// q centrato su 32, scala di gruppo = d * sc[v/16].
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = fp16_to_f32(block[208] as u16 | ((block[209] as u16) << 8));
    if !d.is_finite() {
        out[..256].fill(0.0);
        return;
    }

    let mut yoff = 0usize;
    for n in 0..2 {
        let qloff = n * 64;
        let qhoff = n * 32;
        let scoff = n * 8;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[qloff + l] & 0x0F) | (((qh[qhoff + l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[qloff + l + 32] & 0x0F) | (((qh[qhoff + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) as i32 - 32;
            out[yoff + l] = d * sc[scoff + is] as i8 as f32 * q1 as f32;
            out[yoff + l + 32] = d * sc[scoff + is + 2] as i8 as f32 * q2 as f32;
            out[yoff + l + 64] = d * sc[scoff + is + 4] as i8 as f32 * q3 as f32;
            out[yoff + l + 96] = d * sc[scoff + is + 6] as i8 as f32 * q4 as f32;
        }
        yoff += 128;
    }
}

/// Dequantizza l'intero tensore descritto da `info` con dati grezzi `raw`.
pub fn dequantize(info: &TensorInfo, raw: &[u8]) -> Result<Vec<f32>> {
    let n = info.n_elements() as usize;
    let mut out = vec![0.0f32; n];

    match info.ggml_type {
        GgmlType::F32 => {
            if raw.len() < n * 4 {
                return Err(GgufError::Truncated { expected: n * 4, found: raw.len() });
            }
            for (i, chunk) in raw.chunks_exact(4).take(n).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => {
            if raw.len() < n * 2 {
                return Err(GgufError::Truncated { expected: n * 2, found: raw.len() });
            }
            for (i, chunk) in raw.chunks_exact(2).take(n).enumerate() {
                out[i] = fp16_to_f32(chunk[0] as u16 | ((chunk[1] as u16) << 8));
            }
        }
        GgmlType::Bf16 => {
            if raw.len() < n * 2 {
                return Err(GgufError::Truncated { expected: n * 2, found: raw.len() });
            }
            for (i, chunk) in raw.chunks_exact(2).take(n).enumerate() {
                out[i] = bf16_to_f32(chunk[0] as u16 | ((chunk[1] as u16) << 8));
            }
        }
        GgmlType::Q5_0 => {
            let (bs, tb) = GgmlType::Q5_0.block_layout();
            let mut off = 0usize;
            for block in raw.chunks_exact(tb) {
                if off + bs > n { break; }
                dequant_q5_0(block, &mut out[off..off + bs]);
                off += bs;
            }
        }
        GgmlType::Q8_0 => {
            let (bs, tb) = GgmlType::Q8_0.block_layout();
            let blocks = raw.chunks_exact(tb);
            let mut off = 0usize;
            for block in blocks {
                if off + bs > n { break; }
                dequant_q8_0(block, &mut out[off..off + bs]);
                off += bs;
            }
        }
        GgmlType::Q4K => {
            let (bs, tb) = GgmlType::Q4K.block_layout();
            let mut off = 0usize;
            for block in raw.chunks_exact(tb) {
                if off + bs > n { break; }
                dequant_q4_k(block, &mut out[off..off + bs]);
                off += bs;
            }
        }
        GgmlType::Q6K => {
            let (bs, tb) = GgmlType::Q6K.block_layout();
            let mut off = 0usize;
            for block in raw.chunks_exact(tb) {
                if off + bs > n { break; }
                dequant_q6_k(block, &mut out[off..off + bs]);
                off += bs;
            }
        }
        other => {
            return Err(GgufError::BadTensor(format!(
                "dequant per {:?} non ancora implementata (F0: F32/F16/BF16/Q8_0/Q4_K/Q6_K)",
                other
            )));
        }
    }
    Ok(out)
}
