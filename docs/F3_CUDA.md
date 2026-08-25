# F3 — Backend CUDA: piano di lavoro

Stato: **F3a completata** (astrazione backend, verificata senza GPU).
Prossimo stadio: F3b su hardware reale.

## Il vincolo che detta tutto: la memoria

Il target F3 è **9B Q4_K_M ≥ 40 tok/s su RTX 3090 (24 GB)**.

- Pesi Q4_K_M di un 9B ≈ 5–6 GB → ci stanno comodi.
- Ma il path attuale dequantizza TUTTO in f32 al load → 9B f32 = 36 GB →
  NON entra nella 3090 nemmeno con offload parziale.

Conseguenza architetturale: i pesi quantizzati devono restare compressi sul
device e venire **dequantizzati dentro i kernel di matmul** (fused
dequant-matmul). Non è un'ottimizzazione: è un requisito del target.
llama.cpp fa lo stesso da anni; lo facciamo con kernels nostri (D1).

## Staging

### F3a ✅ — astrazione backend (fatto 25/08, senza GPU)
- Trait `Device` in s-models (`device.rs`): Cache tipizzato per backend,
  `new_cache`, `forward_cached`. Impl CPU su Model (KvCache host).
- `generate` generico + nucleo `generate_tokens` indipendente dal Tokenizer
  (decodifica sempre host-side, qualunque sia il device).
- Test con FakeDevice: il loop prefill→decode→EOS gira attraverso il trait
  senza alcun modello reale. Suite completa verde (41 test).
- s-backend-cuda scaffold: cust dietro feature opzionale `cuda`; workspace
  compilabile ovunque senza toolchain NVIDIA (CI e repo pubblica inclusi).

### F3b — forward f16/f32 su GPU, parity (prima settimana su hardware)
Requisito: accesso SSH al centro-operativo (RTX 3090) o nodo inferenza (2070).
- CudaCtx: init driver, device, stream, errori gestiti (mai panic in serving).
- Upload pesi f32 (0.5B ≈ 2 GB: ok) residente su device; token IDs in, logits out.
- Kernels minimi, UNO ALLA VOLTA con parity contro CPU:
  1. matmul naive f32 (y = x@W^T)
  2. rmsnorm
  3. RoPE NEOX half-split (posizioni assolute — stesso schema di F2.5)
  4. softmax causale per-head (GQA)
  5. silu-mul fused
  6. add residuo
- KV cache su device, stessa semantica posizioni assolute di KvCache.
- **Verifica F3b**: generazione greedy multi-token su 0.5B == output CPU
  al byte (o tolleranza documentata se f16), server end-to-end su GPU.

### F3c — quantizzazione on-device (il salto)
- Q8_0 prima (semplice: blocchi 32, scala fp16): fused dequant-matmul.
- Poi Q4_K_M: super-blocchi, scale fp6/fp8 — il formato del target.
- Layout dei pesi su device RIORGANIZZATO a favore del kernel (non copiare
  il layout GGUF: transposition/padding fatti una tantum al load).
- **Verifica F3c**: 9B Q4_K_M carica nella 3090, genera coerente.

### F3d — performance verso il target
- Matmul tiled + occupancy tuning; benchmark anti-cheat come in F4.
- cuBLASLt FFI SOLO se misuriamo < target con kernels Rust puri (decisione
  D1: fallback esplicito e documentato, non scorciatoia).
- **Verifica F3**: ≥40 tok/s single-stream, 9B Q4_K_M, 3090.

## Kernel inventory (tutti Rust via cust-rs, D1)

| Kernel | Input | Note |
|---|---|---|
| dequant-matmul | W quantizzato residente, x f16/f32 | fused: dequant nel kernel |
| rmsnorm | x [seq×n], w [n] | riduzione per riga |
| rope-neox | q/k [seq×heads×hd], pos assolute | half-split verificato in F1 |
| attn-softmax | scores causali, GQA | fused per-head se possibile |
| silu-mul | gate, up [seq×ff] | elementwise fused |
| add-residuo | a, b | elementwise |
| argmax/sample | logits [vocab] | greedy su device; sampling resta host |

## Aperto / decisioni pendenti
***REMOVED***
 serve setup lì (o password una volta).→ serve setup lì (o password una volta).
2. nodo-inferenza (2070) era offline stamattina: riaccenderlo se serve come
   secondo banco CUDA (sm_75, buona copertura compatibilità).
3. Architettura compute della 3090 = sm_86: verificare versione CUDA toolkit
   installata lì prima di scegliere target PTX.
