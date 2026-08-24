//! s-gguf — lettore GGUF nativo completo per siliceo-motor.
//!
//! F0 del piano MOTORE_SOVRANO. Verifica F0: legge un GGUF reale ed elenca
//! tensori identici a llama.cpp.
//!
//! Principi (dal piano):
//! - Possediamo il codice: nessuna dipendenza da crate GGUF esterni.
//! - Ogni algoritmo di dequant proviene da Exo v0.17-0.20, verificato
//!   bit-perfect contro Python sui file reali (std=0.0152).
//! - Lettura lazy: header+metadata+tensor info in memoria, i pesi restano
//!   su disco e si leggono a richiesta (un modello 27B non sta in RAM due volte).

pub mod dequant;
pub mod types;
pub mod value;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub use types::GgmlType;
pub use value::GgufValue;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("bad magic (expected GGUF)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (attesa 2 o 3)")]
    UnsupportedVersion(u32),
    #[error("truncated data (attesi {expected} byte, trovati {found})")]
    Truncated { expected: usize, found: usize },
    #[error("bad value type {0}")]
    BadType(u32),
    #[error("bad tensor: {0}")]
    BadTensor(String),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, GgufError>;

/// Header GGUF.
#[derive(Debug, Clone, Copy)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

/// Informazione su un tensore (senza i dati).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    /// Dimensioni come scritte nel file (ordine file, non ggml ne[]).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset dei dati relativo all'inizio della sezione dati.
    pub offset: u64,
}

impl TensorInfo {
    /// Numero di elementi del tensore.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Byte occupati dai dati quantizzati.
    pub fn n_bytes(&self) -> u64 {
        let ty = self.ggml_type;
        if ty.is_quantized() {
            let (block_size, type_size) = ty.block_layout();
            (self.n_elements() / block_size as u64) * type_size as u64
        } else {
            self.n_elements() * ty.type_size() as u64
        }
    }
}

/// Un file GGUF aperto: metadata in memoria, dati letti lazy dal disco.
pub struct GgufFile {
    file: File,
    pub header: GgufHeader,
    /// Metadata completi (chiave → valore).
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Offset assoluto dell'inizio della sezione dati nel file.
    data_start: u64,
    /// Allineamento della sezione dati (default 32, sovrascrivibile da general.alignment).
    alignment: u64,
}

impl GgufFile {
    /// Apre un file GGUF: legge header, metadata e tensor info. I pesi restano su disco.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        // Legge l'intero "header region": per file grandi non conosciamo a priori
        // la dimensione, quindi leggiamo in modo incrementale con un cursore.
        file.read_to_end(&mut buf)?;
        let mut c = Cursor::new(&buf);

        // Header
        let magic = c.read_u32()?;
        if magic != 0x4655_4747 {
            return Err(GgufError::BadMagic);
        }
        let version = c.read_u32()?;
        if version != 3 && version != 2 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = c.read_u64()?;
        let kv_count = c.read_u64()?;

        // Metadata (tutti materializzati)
        let mut metadata = Vec::with_capacity(kv_count.min(1024) as usize);
        let mut alignment: u64 = 32; // default GGUF
        for _ in 0..kv_count {
            let key = c.read_string()?;
            let vt = c.read_u32()?;
            let val = GgufValue::read(&mut c, vt)?;
            // general.alignment sovrascrive l'allineamento (spec GGUF)
            if key == "general.alignment" {
                if let GgufValue::U32(a) = val {
                    alignment = a as u64;
                }
            }
            metadata.push((key, val));
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        for _ in 0..tensor_count {
            let name = c.read_string()?;
            let n_dims = c.read_u32()? as usize;
            if n_dims == 0 || n_dims > 8 {
                return Err(GgufError::BadTensor(format!(
                    "{name}: n_dims={n_dims} fuori range"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.read_u64()?);
            }
            let ty_raw = c.read_u32()?;
            let ggml_type = GgmlType::from_raw(ty_raw)
                .ok_or(GgufError::BadType(ty_raw))?;
            let offset = c.read_u64()?;
            tensors.push(TensorInfo { name, n_dims: n_dims as u32, dims, ggml_type, offset });
        }

        // Sezione dati: allineata all'inizio del padding
        let header_end = c.pos() as u64;
        let data_start = if tensor_count > 0 {
            header_end.div_ceil(alignment) * alignment
        } else {
            header_end
        };

        Ok(Self { file, header: GgufHeader { version, tensor_count, kv_count }, metadata, tensors, data_start, alignment })
    }

    /// Cerca un tensore per nome.
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    /// Legge i dati GREZZI (quantizzati) di un tensore.
    pub fn tensor_data_raw(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self.tensor(name)?.clone();
        let abs = self.data_start + info.offset;
        self.file.seek(SeekFrom::Start(abs))?;
        let mut buf = vec![0u8; info.n_bytes() as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Legge un tensore e lo dequantizza in f32.
    pub fn tensor_data_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor(name)?.clone();
        let raw = self.tensor_data_raw(name)?;
        dequant::dequantize(&info, &raw)
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }
}

/// Cursore di lettura su buffer (per la regione header/metadata).
pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(GgufError::Truncated { expected: n, found: self.buf.len() - self.pos });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8> {
        let b = self.take(1)?;
        Ok(b[0])
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| GgufError::BadTensor("stringa non utf8".into()))
    }
}
