# D6 — Configurazione Dinamica: un motore per tutto

**Data:** 2026-08-25 · **Stato:** Design approvato
**Direttiva di progetto:** *"il motore deve raccogliere il meglio degli altri anche mettendo
un metodo di configurazione dinamica. devono usarne uno per tutto."*

---

## 1. Cosa significa, in concreto

Oggi il server si configura SOLO da riga di comando (`s-server modello.gguf --port N`).
Per cambiare qualunque cosa si deve spegnere e riavviare. Chi arriva da Ollama o llama.cpp
si aspetta:

1. **File di configurazione** — le scelte persistono, non si ridigitano flag ogni volta
2. **Cambi a runtime** — temperatura/porta/default cambiano senza ricompilare né riavviare
3. **Auto-rilevamento** — l'architettura del modello, i thread, il template di chat si capiscono da soli
4. **Stesso binario ovunque** — CPU, CUDA, Vulkan: la scelta è configurazione, non compilazione

## 2. Stato attuale del codice (verificato)

| Componente | Dove | Config oggi |
|---|---|---|
| Server HTTP | s-server/main.rs | solo argv: path modello, --port |
| Parametri generazione | s-models/generate.rs `GenerateParams` | default hardcoded, override per-request dal JSON |
| Arch modello | s-models/qwen2.rs `Config::from_gguf` | letta dal metadata GGUF (già dinamica ✅) |
| Template chat | s-server/main.rs `render_chatml` | HARDCODED ChatML Qwen ⚠️ |
| Thread/CPU | n/d | single-thread per request, forward scalare |

Già buono: l'arch del modello è letta dal GGUF. Il gap vero è: template chat cablato,
niente file di config, niente modifiche a runtime, niente cambio modello senza riavvio.

## 3. Design: config a strati

Precedenza dal più debole al più forte (l'ultimo vince):

```
1. DEFAULT      dentro il codice (greedy, 256 token, porta 8096)
2. SISTEMA      /etc/siliceo-motor/motor.json   (opzionale, deploy macchine)
3. UTENTE       ./motor.json oppure --config PATH
4. CLI          flag sulla riga di comando (--port, --model, ...)
5. REQUEST      parametri nel body della singola richiesta (già esiste)
6. RUNTIME      POST /v1/config → aggiorna lo stato vivo del server
```

Regola: ogni strato specifica SOLO ciò che vuole cambiare; il resto eredita dal sotto.

### Struttura del file (JSON — parser nostro già esiste, zero dipendenze nuove)

```json
{
  "server":  { "host": "127.0.0.1", "port": 8096, "threads": 0 },
  "model":   { "path": "modelli/qwen2.5-0.5b.gguf",
               "context": 4096,
               "template": "auto" },
  "generate": { "temperature": 0.0, "top_p": null, "top_k": null,
                "max_tokens": 256 },
  "backend":  { "device": "auto", "n_gpu_layers": -1 }
}
```

- `"threads": 0` e `"device": "auto"` = rileva da solo (available_parallelism / probe CUDA)
- `"template": "auto"` = legge il chat template dal GGUF se presente, altrimenti fallback noto per arch

## 4. API runtime (estensione nostra, stile OpenAI)

| Endpoint | Fa |
|---|---|
| `GET /v1/config` | restituisce la config effettiva (tutti gli strati fusi) |
| `POST /v1/config` | patch parziale: `{temperature: 0.7}` cambia solo quello |
| `POST /v1/model` | hot-swap: `{path: "altro.gguf"}` carica e sostituisce il modello a processo vivo |
| `GET /health` | arricchito: arch, device attivo, thread, template |

Il hot-swap usa `ArcSwap<App>`-like fatto in casa: `Arc<RwLock<Arc<App>>>` — le richieste
in corso finiscono col vecchio modello, le nuove prendono il nuovo. Zero downtime.

## 5. Auto-rilevamento ("uno per tutto")

| Cosa | Come |
|---|---|
| Architettura | `general.architecture` dal GGUF → registro arch (`qwen2` oggi, `llama` alias) |
| Template chat | campo `tokenizer.chat_template` nel GGUF (Jinja) → interim: tabella nota per famiglia, Jinja engine in fase successiva |
| Device | compile-time features + probe runtime (cuda disponibile?) |
| Threads | `std::thread::available_parallelism()` |

## 6. Fasi verificabili

### D6a — Config a strati (questa sessione)
- [ ] crate `s-config`: parse JSON ( riusa parser s-server spostato in comune), fusione strati
- [ ] `--config PATH` + auto-discovery `./motor.json`
- [ ] Test: fusione strati, precedenza, valori parziali
- **Verifica**: server parte con file config che cambia porta+default temperatura, curl lo conferma

### D6b — API runtime
- [ ] `GET/POST /v1/config`, health arricchito
- [ ] Test end-to-end curl
- **Verifica**: cambio temperatura a server acceso, risposta cambia comportamento senza restart

### D6c — Hot-swap modello ✅ CHIUSA (commit a592e5f)
- [x] `App` dietro RwLock, `POST /v1/model`
- [x] Gestione errori: path sbagliato → errore pulito, vecchio modello resta vivo
- **Verifica**: Qwen 0.5B ↔ SmolLM2 andata/ritorno a server acceso; bonus: bias opzionali → famiglia llama supportata, parity F1 riverificata

### D6d — Template dinamico ✅ CHIUSA (25/08 notte)
- [x] rilevamento: config > tokenizer.chat_template nel GGUF > architettura (ChatML, Llama3)
- [x] riconoscimento per firma nel template GGUF (Jinja completo NON obiettivo ora)
- [x] BUG SCOPERTO E FIXATO: encode() spezzava i token speciali (<|im_start|> → pezzi BPE sbagliati); ora longest-match prima della regex, test su file reale
- [x] FIX preesistente: EOS produceva finish_reason "length" invece di "stop"
- **Verifica**: stesso server, Qwen e SmolLM chattano con prompt codificato giusto; parity raw intatta

## 7. Decisioni aperte

1. **Formato**: propongo JSON (parser nostro già pronto, zero dipendenze). TOML sarebbe
   più idiomatico ma ci obbliga a scrivere un secondo parser o prendere una crate esterna.
2. **Nome endpoint**: `/v1/config` (stile OpenAI) vs `/api/*` (stile Ollama). Propongo `/v1/*`.
3. **Scopo D6a-D6b questa sessione**, D6c-D6d dopo: ok?

## 8. Non obiettivi (anti scope-creep)

- Niente UI web di configurazione (la dashboard arriva in F7)
- Niente Jinja completo ora (solo tabella famiglie + riconoscimento)
- Niente multi-modello simultaneo (hot-swap = UN modello caricato alla volta)
