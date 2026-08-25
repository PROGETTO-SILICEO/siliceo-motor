# Roadmap

Ogni fase ha una **verifica oggettiva e riproducibile**: niente fase è "chiusa" senza
un test che la dimostri. F0–F2.5 e la configurazione dinamica sono completate e verificate.

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

## ✅ F2.5 — KV cache incrementale
- Cache per layer con posizioni RoPE assolute: prefill una volta, poi un token per passo
- **Verifica**: output identico al byte a llama.cpp; ~5x di velocità sul 0.5B CPU (0.12 → 0.63 tok/s)

## ✅ Configurazione dinamica
- Config a strati con precedenza: fabbrica < `/etc/siliceo-motor/motor.json` < `./motor.json`
  < flag CLI < parametri della richiesta < patch runtime
- `GET/POST /v1/config`: leggere e cambiare i default a server acceso
- Hot-swap del modello via `POST /v1/model`: scambio atomico, richieste in corso mai
  interrotte, errore di caricamento non tocca il modello attivo
- Template di chat rilevati dal GGUF (ChatML, Llama3) con override manuale
- Fix strutturale: token speciali `<|...|>` riconosciuti interi prima della
  pre-tokenizzazione (prima venivano spezzati e ricodificati BPE con id sbagliati)
- Supporto famiglia llama (bias QKV opzionali nel loader)
- **Verifica**: swap Qwen 0.5B ↔ SmolLM2 135M a server acceso; patch runtime applicata
  alla generazione successiva; parity logits invariata dopo ogni modifica

## 🚧 F3 — Backend CUDA (kernel scritti in Rust, compilati a PTX/cubin)
- ✅ Astrazione `Device`: forward e generazione backend-agnostici (CPU oggi, CUDA/Vulkan domani)
- ✅ Pipeline kernel sovrana operativa: Rust → rustc_codegen_nvvm → PTX → ptxas → cubin
  **precompilato a build time** (niente JIT del driver all'avvio: avvio deterministico)
- ✅ Kernel f32 con parity GPU/CPU misurata a 1–2 ulp: matmul, RMSNorm, RoPE NEOX,
  softmax, SwiGLU (silu-mul), add residuo — matematica via crate `libm`,
  rimappata automaticamente agli intrinsics libdevice dal codegen NVVM
- 🔜 Pesi e KV cache residenti su device, kernel di attenzione fused GQA,
  forward completo con parity token-identica vs CPU
- 🔜 Quantizzazione on-device: fused dequant-matmul sui K-quants
  (i pesi Q4_K_M restano compressi in VRAM — requisito, non ottimizzazione)
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
