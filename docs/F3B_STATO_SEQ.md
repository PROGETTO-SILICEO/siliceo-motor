# F3b finestra 1 — Stato sequenza generalizzato, registro famiglie, agganci futuri

Data: 2026-08-26 · Decisione roadmap v2 del 25/08

## Perché ORA

Rifattorizzare stato+trait+runtime DOPO che server/hot-swap/generate toccano
il modello = giorni. Anticipare l'astrazione prima di chiudere F3b step 4 = ore.
Il modello di punta (Qwen3.8-27B) è IBRIDO DeltaNet [KB-01,05]: lo stato KV-denso
per tutti i layer è un'assunzione destinata a rompersi.

## Cosa è cambiato (commit di questa finestra)

1. **`s-models/src/state.rs`** — `SeqState` sostituisce `kv::KvCache`:
   - `LayerSeq::Kv(LayerKv)` — attenzione con K/V `[max_seq × dim]`
   - `LayerSeq::Recurrent(RecurrentState)` — buffer a dimensione fissa,
     indipendente da max_seq (DeltaNet). Layout dati definitivo = decisione F5.
   - `SeqState::all_kv(...)` = caso dense (qwen2/llama), scorciatoia per l'99%
   - `SeqState::from_layers(vec![...])` = caso MISTO (es. 16 layer attention
     con KV + 48 ricorrenti)
   - `reset()`: KV → solo `len=0` (zona oltre len mai letta); Recurrent →
     azzeramento vero (porta memoria tra token).
   - Accesso al variant sbagliato = panic con "dispatch architettura errato":
     errore di dispatch del modello, mai silenzioso.
   - API storica (`layer_k_mut`, `layer_k_upto`, `advance`, `len`) invariata:
     qwen2::forward_cached NON è stato toccato (parity riverificata).

2. **`s-models/src/registry.rs`** — enum `Famiglia`, detection da
   `general.architecture` GGUF. Oggi: solo `Qwen2Llama` (arch "qwen2"/"llama").
   Rifiuto pulito e onesto su architetture non supportate. Il mapping
   enum→loader cresce SOLO qui, mai sparsa tra server/hot-swap/generate.
   SmolLM NON mappato finché non verificato con parity (D6c usa regex GPT-2
   approssimate nel tokenizer: prima la parity, poi il mapping).

3. Trait `Device`: INVARIATO nella forma (`type Cache` era già astratto).
   Cambiato solo il tipo concreto dell'impl CPU (`Cache = SeqState`).

## Verifica (fatta, non data per scontata)

- Suite workspace completa verde, zero warning
- Parity logits F1 col 0.5B reale (`--ignored --release`): top-6 identico ✅
- E2E generate attraverso il trait: "La capitale dell'Italia è" →
  " la capitale dell'Italia, ma non è la capit" == riferimento F2 al byte ✅

## Punto di aggancio MULTIMODALE (mmproj) — PROGETTATO, NON IMPLEMENTATO

Pattern ecosystem [KB-05]: il tower visivo viaggia in un GGUF SEPARATO
(mmproj) e si aggancia a monte degli embedding del LLM testuale:

```
immagine → encoder vision (mmproj) → proiezione nello spazio embd
        → token "embedding" INIETTATI nella sequenza al posto di token ID
        → forward testuale invariato da lì in poi
```

Conseguenze sull'astrazione attuale (da rispettare quando si implementa):
1. L'iniezione avviene TRA tokenizer e forward: il trait Device non vede
   nulla di multimodale. `forward_cached` riceve embeddings? NO — oggi riceve
   `&[u32]`. Il punto di estensione è il MODELLO, non il device: servirà un
   modo per passare righe-embedding pre-calcolate al posto del lookup
   `tok_embd[token]` (es. input enum `{ Id(u32), Embedding(Vec<f32>) }` o una
   variante con matrice opzionale a fianco degli id). DECIDERE quando si
   tocca il codice, non prima.
2. Lo stato di sequenza NON cambia: le posizioni immagine occupano posizioni
   RoPE come i token testo; la cache cresce uguale.
3. Il registry Famiglia crescerà di una dimensione: modelli mmproj accoppiati
   (coppia main-gguf + mmproj-gguf validata a load).

## Decode pluggable — PROGETTATO, NON IMPLEMENTATO

Oggi il decode è autoregressivo puro: `generate_tokens` (generate.rs) È la
strategia AR. La diffusione (DiffusionGemma [KB]) rompe il loop
un-token-per-passi:

- input: prompt completo + budget di iterazioni
- output: sequenza di ids (non necessariamente crescente di 1)

Contratto per rendere la strategia pluggabile SENZA rifacimento:
1. `GenerateParams` resta il contenitore comune (max_tokens, stop, seed).
2. La strategia produce `Vec<u32>` + `finish_reason`; stop strings e decodifica
   testo restano host-side, fuori dalla strategia (già così: `decode` è closure).
3. `Device::forward_cached(tokens, &mut SeqState)` è già sufficiente anche per
   diffusione: processa N token per chiamata e ritorna i logits dell'ultimo.
   Eventuali logits PER-POSIZIONE (servono alla diffusione per revisionare più
   token insieme) saranno un metodo addizionale del trait, aggiunto quando
   serve davvero — non anticipato a vuoto.
4. Il sampler non cambia: greedy/temp/top-k/top-p operano su logits, qualunque
   sia la strategia che li ha prodotti.

## Prossimi passi

- [ ] F3b step 4: pesi residenti su device (upload f32 una tantum per 0.5B;
      fused dequant-matmul vincolante per 9B — 36GB f32 non entrano in 24GB)
- [ ] KV cache su device (buffer max_seq, append K/V post-RoPE)
- [ ] Kernel attention fused (dot q·k + scale + softmax prefix + Σw·v, GQA)
- [ ] Forward qwen2 completo su GPU, parity token-identica vs CPU/llama.cpp
