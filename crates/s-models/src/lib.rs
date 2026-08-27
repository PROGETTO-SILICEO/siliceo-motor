//! s-models — architetture di modelli per siliceo-motor.
//!
//! F1: forward pass CPU naive f32 per l'architettura qwen2 (== llama con bias
//! QKV). Obiettivo: PARITY dei logits con llama.cpp sullo stesso prompt.

pub mod chat;
pub mod device;
pub mod generate;
pub mod qwen2;
pub mod qwen35;
pub mod registry;
pub mod sampling;
pub mod state;

pub use chat::ChatTemplate;
pub use device::Device;
pub use qwen2::{Config, Model, WeightViews};
pub use qwen35::Config as Qwen35Config;
pub use qwen35::Model as Qwen35Model;
pub use registry::Famiglia;
pub use sampling::Sampler;
pub use state::SeqState;
