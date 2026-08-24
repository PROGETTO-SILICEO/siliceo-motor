# Roadmap

Ogni fase ha una **verifica oggettiva e riproducibile**: niente fase è "chiusa" senza
un test che la dimostri. F0–F2 sono completate e verificate.

## ✅ F0 — Fondazioni GGUF
- Parser GGUF v3 completo: metadata (tutti i 13 tipi), tensor table, regioni dati
- Dequantizzazione K-quants: Q4_K, Q5_0, Q6_K, Q8_0 verificati contro gguf-py ufficiale
- **Verifica**: 291 tensori al byte identici a `llama-gguf` su file reale

## ✅ F1 — Tokenizer + forward CPU
- BPE byte-level nativa dal GGUF (vocabolario + merges + pre-tokenizer regex Qwen2)
- Forward qwen2 CPU f32: RMSNorm, bias QKV, GQA, RoPE NEOX, SwiGLU
- **Verifica**: token IDs identici a llama.cpp; generazione greedy token-identica; top-6 logprob combaciante

## ✅ F2 — Sampling + server end-to-end
- Sampler greedy / temperature / top-k / top-p con RNG seedabile (splitmix64)
- Loop autoregressivo: max_tokens, EOS multipli, stop strings
- Server OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`) su std::net,
  zero dipendenze HTTP/JSON
- **Verifica**: curl end-to-end; parity mantenuta attraverso l'intero stack

## 🔜 F2.5 — KV cache incrementale
- Cache per layer con rollback per speculative decoding
- Salto di performance misurabile anche su CPU (fine del full-recompute)

## 🔜 F3 — Backend CUDA (kernels in Rust via cust-rs)
- Gemm f16/bf16/int8, KV cache su device, offload a layer parziale → totale
- Filosofia: kernel posseduti, non wrapper. FFI cuBLASLt solo come fallback documentato
- **Target**: 9B Q4_K_M ≥ 40 tok/s single-stream su RTX 3090

## 🔜 F4 — Performance parity
- Kernel fused (RoPE+attention), batching decode multi-sequence
- Benchmark suite anti-cheat come regression gate
- **Target**: ≥80% di llama.cpp su dense 8B Q4

## 🔜 F5 — Architetture ibride SSM
- Stati ricorrenti lineari + full attention ogni N layer (stile Qwen3.5)

## 🔜 F6 — Speculative decoding corretto + MTP nativo
- Rejection sampling con distribuzioni complete, accesso diretto agli hidden state

## 🔜 F7 — Multi-utente
- Continuous batching, scheduler, metriche
