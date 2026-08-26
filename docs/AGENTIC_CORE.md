# AGENTIC CORE — progetto (fase 3 roadmap v2)

Data: 2026-08-26 · Decisione roadmap v2 (promozione al posto 3): "tutto è agentic ora,
anche tu". Il carico dominante 2026 è agentico ; un serving
senza tool calling è autoesclusione. NON è feature-race: tocca il MOTORE
(sampling vincolato, prefix caching, streaming parziale).

## Le tre capacità

### AC-1 Streaming SSE (`stream: true`)
Il client riceve i token man mano che nascono, non a fine generazione.
- API di generazione: `generate_tokens` guadagna una callback
  `on_token: &dyn FnMut(u32) -> bool` (false = stop lato server, per abort client).
  UN SOLO loop interno: niente duplicazione del percorso verificato.
- Trasporto: HTTP/1.1 `Content-Type: text/event-stream` + `Connection: close`
  (delimitazione da EOF: compatibile con curl e ogni client OpenAI, zero
  chunked encoding da implementare a mano).
- Formato chunk OpenAI: primo chunk con `delta:{role}`, chunk `delta:{content}`
  per ogni PEZZO DI TESTO NUOVO (diff sul decode dell'intera sequenza: gestisce
  UTF-8 spezzato tra token), ultimo chunk con `finish_reason`, poi `data: [DONE]`.
- Il testo inviato è già filtrato dalle stop conditions (stessa semantica F2).

### AC-2 Tool calling
- Richiesta: campo `tools:[{type:function, function:{name,description,parameters}}]`,
  `tool_choice`. Messaggi aggiuntivi: assistant con `tool_calls`, role `tool`
  con `tool_call_id`.
- Prompt: sezione strumenti renderizzata nel formato nativo del modello quando
  esiste (Qwen=hermes-style), altrimenti istruzioni documentate nel sistema.
- Rilevamento chiamata nell'output: tag espliciti `<tool_call>{json}</tool_call>`
  (convenzione ecosystem, robusta anche senza constrained decoding).
  La risposta porta `message.tool_calls:[{id,type,function:{name,arguments}}]`
  e `finish_reason:"tool_calls"`.
- Loop agentic RESTA LATO CLIENT: il server esegue modelli, non azioni.

### AC-3 Structured output (sampling vincolato)
- `response_format:{type:json_object, schema}` → maschera dei logit per passo:
  token ammessi = quelli coerenti con lo stato della grammatica JSON dato il
  prefisso emesso. Serve mappa vocab→stringhe (già in s-tokenizer).
- Stato grammatica: parser JSON incrementale scritto in casa (coerente con lo
  spirito zero-deps): stack contesti (object/array/stringa/valore), set di
  token legali per stato. Primo sottoinsieme: JSON valido qualunque;
  poi vincoli schema (tipi/required).
- Punto di aggancio: `Sampler::sample(logits)` guadagna `mask: Option<&[bool]>`
  (o set di id vietati) applicata PRIMA di temperatura/sampling — la maschera
  è strategia, il sampling resta quello verificato.

### AC-4 Prefix caching ✅ COMPLETATO E VERIFICATO LIVE (26/08)

   - server mantiene una piccola LRU (4 voci, `s-server/src/prefix.rs`) di snapshot SeqState
     clonati SUBITO DOPO il prefill (callback `on_prefill` di `generate_tokens_seeded`)
   - riuso = il token-vettore dello snapshot è PREFISSO ESATTO dei token richiesti
     (confronto al byte, nessun hash che può collidere); vince il prefisso più lungo
   - `SeqState` ora ha `Clone`, `capacity()`, `truncate(n)` (Err su stati con layer
     ricorrenti: non possono tornare indietro — F5 deciderà il riuso esatto),
     `ensure_capacity(n)` (crescita preservando i dati scritti)
   - **ESTENSIONE DI CAPACITÀ AL VOLO** (scoperta dal test live, invisibile agli unit
     test): lo snapshot è dimensionato sulla richiesta precedente; la conversazione
     dell'agente CRECE → al riuso lo stato cresce fino alla capacità richiesta.
     Senza questo il caso d'uso principale (conversazioni agentiche) mancava la cache
   - `generate_tokens_seeded`: contratto rigoroso — chi semina garantisce
     `cache.len() == n` e `n < prompt.len()`; il delta-prefill produce logits IDENTICI
     per causalità. `Generated.prefix_reused` riporta le posizioni risparmiate
   - invalidazione: la cache vive in `Inner` → l'hot-swap (D6c) la scarta tutta
   - VERIFICA LIVE (qwen2.5-0.5B, seed fisso): identica → riuso 53/54 e output
     identico; conversazione estesa → riuso 54/89 con estensione capacità;
     server FREDDO == CALDO al byte; streaming non regredisce. Suite: 81 test,
     3 run stabili, zero warning
   - BUG COLLEGATO FIXATO (preesistente): due test s-gguf scrivevano lo stesso file
     temp → gara tra test paralleli (EOF casuale)

## Ordine delle fette verticali
1. ✅ AC-1 streaming (commit 0b2878b) — verificato live
2. ✅ AC-2 tool calling (commit 2ff13ba) — round trip completo verificato live
3. ✅ AC-3 maschera grammaticale JSON (commit 2ff13ba) — 4/4 validi live, streaming incluso
4. ✅ AC-4 prefix caching — COMPLETATO E VERIFICATO (26/08, vedi sezione qui sopra)

## Non-fatti (scelti)
- Niente esecuzione di tool lato server (il server non è un agente).
- Niente dipendenze esterne per SSE/JSON/grammar: std::net + codice nostro.
