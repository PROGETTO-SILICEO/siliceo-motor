//! Valori dei metadata GGUF: tutti i 13 tipi della specifica, materializzati.
//!
//! Codici tipo (da gguf.h):
//! 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool, 8=string,
//! 9=array, 10=u64, 11=i64, 12=f64

use super::{Cursor, GgufError, Result};

/// Valore di metadata GGUF.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Legge un valore del tipo `vt` dal cursore. (Interno: `Cursor` è pub(crate).)
    pub(crate) fn read(c: &mut Cursor, vt: u32) -> Result<Self> {
        Ok(match vt {
            0 => Self::U8(c.read_u8()?),
            1 => Self::I8(c.read_u8()? as i8),
            2 => Self::U16(c.read_u16()?),
            3 => Self::I16(c.read_u16()? as i16),
            4 => Self::U32(c.read_u32()?),
            5 => Self::I32(c.read_u32()? as i32),
            6 => Self::F32(f32::from_bits(c.read_u32()?)),
            7 => Self::Bool(c.read_u8()? != 0),
            8 => Self::String(c.read_string()?),
            9 => {
                let elem_type = c.read_u32()?;
                let n = c.read_u64()?;
                // Protezione: array enormi di tipi piccoli sono legittimi
                // (es. tokenizer vocabolario), ma un count assurdo indica corruzione.
                if n > 100_000_000 {
                    return Err(GgufError::BadTensor(format!("array con count {n}")));
                }
                let mut v = Vec::with_capacity(n.min(1_000_000) as usize);
                for _ in 0..n {
                    v.push(Self::read(c, elem_type)?);
                }
                Self::Array(v)
            }
            10 => Self::U64(c.read_u64()?),
            11 => Self::I64(c.read_u64()? as i64),
            12 => Self::F64(f64::from_bits(c.read_u64()?)),
            other => return Err(GgufError::BadType(other)),
        })
    }

    /// Helper: valore come u64 (per dimensioni, conteggi).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::I8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::I16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::I32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I64(v) => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }
}
