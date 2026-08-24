# siliceo-motor

Motore di inferenza universale in Rust nativo — Progetto Siliceo.

Stesso binario su RTX 3090 (Centro Operativo), RTX 2070 (nodo inferenza), GTX 1650 / Windows (PC-Lele) e ovunque serva.

## Principi

- **Sovrano**: nessuna dipendenza C++ obbligatoria. I backend C restano plugin opzionali.
- **Universale**: Linux/Windows/macOS, backends CPU/CUDA/Vulkan.
- **API-first**: server OpenAI-compatible dal giorno uno.
- **Empirico**: ogni fase ha una verifica misurabile. Niente "completato" senza prova.
- **Studia, copia, migliora**: sovrano significa possedere il codice, non reinventarlo.

## Stato

Vedi `docs/` e il piano di riferimento. Fasi: F0 fondazioni → F1 forward CPU → F2 sampling → F3 CUDA → F4 performance → F5 SSM ibrido → F6 MTP nativo → F7 produzione.

| Crate | Ruolo | Stato |
|---|---|---|
| s-gguf | Lettore GGUF completo + dequant F32/F16/BF16/Q8_0/Q4_K/Q6_K | ✅ F0: verificato su file reale (272 tensori, dequant sana) |
| s-types, s-tokenizer, s-backend*, s-models, s-kv, s-speculative, s-server, s-voice | scaffold | 🚧 F0 in corso |
