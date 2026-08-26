# siliceo-motor

**Un motore di inferenza LLM sovrano in Rust nativo** — dal formato GGUF al server OpenAI-compatible, senza wrapper, senza binding C++, senza dipendenze HTTP/JSON.

```
curl → s-server → JSON (in casa) → tokenizer BPE nativa → forward f32 → sampler → risposta OpenAI-compatible
```

## Perché

I motori dominanti sono ottimi, ma usarli significa delegare: kernel C++, graph capture,
licenze AGPL, stack Python. *Sovrano* vuol dire possedere il codice — ogni livello di
questo motore è scritto da noi e verificabile:

- **s-gguf**: parser GGUF v3 completo con dequantizzazione K-quants
- **s-tokenizer**: BPE byte-level che legge vocabolario e merges direttamente dal GGUF
- **s-models**: architetture (qwen2/llama) con forward CPU f32 + KV cache incrementale
- **s-backend-cuda**: kernel scritti in Rust (Rust → PTX → cubin, niente JIT a runtime)
- **s-server**: HTTP/1.1 su `std::net`, zero dipendenze esterne

## Stato: parity verificata contro llama.cpp

Non è un prototipo "sembra funzionare": ogni fase è chiusa con verifica oggettiva.

| Fase | Componente | Verifica |
|---|---|---|
| F0 | s-gguf | **291 tensori al byte** identici a `llama-gguf` su Qwen2.5-0.5B Q4_K_M |
| F1 | s-tokenizer | IDs **identici** a `llama-tokenize` |
| F1 | forward qwen2 | generazione greedy **token-identica** a llama.cpp; top-6 logprob combaciante |
| F2 | sampling + server | endpoint OpenAI-compatible via curl; greedy/top-k/top-p seedabili; stop conditions |
| F2.5 | s-kv | KV cache incrementale: output identico al full-recompute, riuso prefisso verificato live |
| F3b | backend CUDA | kernel f32 con parity GPU/CPU a 1–2 ulp; forward greedy **GPU == CPU al token** su RTX 3090 e 2070 |

### Agentic Core (AC-1..4) — completo

Il carico dominante è agentico: il serving nasce per i tool, non li aggiunge dopo.

- **AC-1 Streaming SSE**: `stream: true`, output byte-identico al non-streaming (seed fisso), stop zero-leak
- **AC-2 Tool calling**: conversione schema JSON → sampling vincolato
- **AC-3 Structured output**: risposta conforme allo schema, garantita
- **AC-4 Prefix caching**: snapshot LRU post-prefill, riuso del prefisso esatto con estensione capacitiva lazy — la seconda richiesta di una conversazione agente riusa decine di token di calcolo

### Riprodurre la parity

```bash
# tokenizer parity
S_TOKENIZER_QWEN05B=model.gguf cargo test -p s-tokenizer --test parity

# logits parity (top-6 identico a llama.cpp)
S_MODELS_QWEN05B=model.gguf cargo test -p s-models --release --test parity_logits -- --ignored

# GGUF parser parity
S_GGUF_QWEN05B=model.gguf cargo test -p s-gguf --test parity_llamacpp
```

## Quick start

```bash
cargo build --release -p s-server

# serve un modello GGUF (testato con Qwen2.5-0.5B-Instruct Q4_K_M)
./target/release/s-server ./model.gguf --port 8096

curl http://127.0.0.1:8096/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Ciao!"}],"max_tokens":50}'
```

### Parametri supportati

`temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop` (stringa o array).
Greedy con `temperature: 0` — deterministico al token.

## Onestà sulle performance

La correttezza viene prima della velocità, sempre: ogni salto di performance è chiuso
solo quando è **verificato contro riferimento indipendente** (parity bit-per-bit o al token).
La KV cache incrementale e i kernel CUDA sono entrati in quest'ordine. I prossimi salti:
kernel fused, quantizzazione on-device (dequant-fused sui K-quants) e architetture ibride
SSM — vedi [ROADMAP.md](ROADMAP.md).

## Licenza

MIT OR Apache-2.0 (a scelta dell'utente), come l'ecosistema Rust.
