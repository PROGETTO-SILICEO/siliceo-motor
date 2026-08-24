//! s-models — architetture di modelli per siliceo-motor.
//!
//! F1: forward pass CPU naive f32 per l'architettura qwen2 (== llama con bias
//! QKV). Obiettivo: PARITY dei logits con llama.cpp sullo stesso prompt.

pub mod generate;
pub mod qwen2;
pub mod sampling;

pub use qwen2::{Config, Model};
pub use sampling::Sampler;
