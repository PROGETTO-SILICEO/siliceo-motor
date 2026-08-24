//! Tipi GGML e layout dei blocchi di quantizzazione.
//!
//! Tabella da ggml.h / ggml-quants.h. I layout verificati empiricamente
//! (Q4_K, Q8_0, Q6_K) provengono da Exo v0.17-0.20, bit-perfect contro Python.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
    Bf16,
    Unknown(u32),
}

impl GgmlType {
    /// Dal codice numerico nel file GGUF.
    pub fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::Iq1M,
            30 => Self::Bf16,
            other => return Some(Self::Unknown(other)),
        })
    }

    /// Dimensione in byte di un singolo valore non quantizzato.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 | Self::Bf16 => 4, // BF16 si espande a f32
            Self::F16 => 2,
            _ => 4,
        }
    }

    /// Per i tipi quantizzati: (elementi per blocco, byte per blocco).
    /// Per i tipi non quantizzati ritorna None (usa `type_size`).
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 256),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq4Nl => (256, 136),
            Self::Iq3S => (256, 102),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Iq1M => (256, 56),
            _ => (32, 4),
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16 | Self::Unknown(_))
    }

    /// Nome ggml canonico (per confronto con llama.cpp --dump).
    pub fn name(&self) -> String {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::Iq1M => "IQ1_M",
            Self::Bf16 => "BF16",
            Self::Unknown(v) => return format!("UNKNOWN({v})"),
        }
        .to_string()
    }
}
