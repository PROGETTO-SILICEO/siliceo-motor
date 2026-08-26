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
Hardware: banco di prova RTX 3090 24GB, CUDA 12.4.

FATTO (25/08):
- ✅ CudaCtx (init driver, contesto primario, describe per /health)
- ✅ Pipeline D1 SOVRANA OPERATIVA: kernel Rust → rustc_codegen_nvvm → PTX
  → ptxas → cubin → cust launch. Nightly pinnato 2026-04-02.
- ✅ Kernel Rust `vec_add_f32` e `matmul_wt_f32` in kernels/src/lib.rs
- ✅ PARITY matmul GPU vs CPU verificata su 3090 (max_err < 1e-3,
  stessa semantica di matmul_rows di s-models)
- Scoperte ambientali documentate nel ledger: driver senza JIT PTX
  (→ precompilazione obbligatoria), find_cuda_helper/CUDA_PATH quirks,
  cuda_std non utilizzabile (half+std) → dipendiamo SOLO da cuda_std_macros
  e usiamo gli intrinsics nativi core::arch::nvptx.

PROSSIMO (restante F3b):
- [ ] rmsnorm kernel + parity
- [ ] RoPE NEOX half-split con posizioni assolute + parity
- [ ] softmax causale GQA fused + parity
- [ ] silu-mul fused, add residuo + parity
- [ ] Pesi residenti su device (0.5B f32), KV cache su device
- [ ] forward completo qwen2 su GPU == output CPU al byte
- **Verifica F3b**: generazione greedy multi-token su 0.5B identica via server

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
1. **Accesso al secondo nodo GPU**: chiave non ancora autorizzata → serve setup.
2. nodo-inferenza (2070) era offline stamattina: riaccenderlo se serve come
   secondo banco CUDA (sm_75, buona copertura compatibilità).
3. Architettura compute della 3090 = sm_86: verificare versione CUDA toolkit
   installata lì prima di scegliere target PTX.
