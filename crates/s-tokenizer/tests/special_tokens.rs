//! Test token speciali — bug D6d: `<|im_start|>` veniva spezzato dalla
//! pre-tokenizzazione e ricodificato via BPE → ID sbagliati nel prompt chat.
//!
//! Richiede un GGUF reale: `S_TOKENIZER_GGUF=/path/modello.gguf cargo test -p s-tokenizer`
//! (senza variabile i test si SALTANO e lo dicono in output, non finto ok).

use s_tokenizer::Tokenizer;

fn tokenizer_reale() -> Option<Tokenizer> {
    let path = match std::env::var("S_TOKENIZER_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SALTATO: imposta S_TOKENIZER_GGUF per eseguire i test su file reale");
            return None;
        }
    };
    Some(Tokenizer::from_gguf(path).expect("tokenizer dal GGUF"))
}

#[test]
fn speciale_intero_non_spezzato() {
    let Some(tok) = tokenizer_reale() else { return };
    let id_atteso = tok.token_to_id("<|im_start|>").expect("<|im_start|> nel vocab");
    let ids = tok.encode("<|im_start|>").unwrap();
    assert_eq!(ids, vec![id_atteso], "<|im_start|> deve dare ESATTAMENTE il suo id");
}

#[test]
fn speciale_in_mezzo_al_testo() {
    let Some(tok) = tokenizer_reale() else { return };
    let im_end = tok.token_to_id("<|im_end|>").unwrap();
    let ids = tok.encode("ciao<|im_end|>mondo").unwrap();
    assert!(ids.contains(&im_end), "l'id di <|im_end|> deve comparire intero");
    // e il testo normale attorno è codificato
    assert!(ids.len() > 3);
}

#[test]
fn roundtrip_con_speciali() {
    let Some(tok) = tokenizer_reale() else { return };
    let ids = tok.encode("a<|im_start|>b").unwrap();
    let testo = tok.decode(&ids).unwrap();
    // <|im_start|> è speciale: la decode lo salta, il resto torna
    assert_eq!(testo, "ab");
}

#[test]
fn longest_match_vince() {
    let Some(tok) = tokenizer_reale() else { return };
    // se esiste <|x|> e <|xy|>, il più lungo vince quando entrambi combaciano
    let lunghi: Vec<&String> =
        tok.specials().iter().filter(|s| s.len() > 8).take(1).collect();
    if lunghi.is_empty() {
        eprintln!("SALTATO: nessun token speciale lungo nel vocabolario");
        return;
    }
    let sp = lunghi[0].clone();
    let id = tok.token_to_id(&sp).unwrap();
    let ids = tok.encode(&sp).unwrap();
    assert_eq!(ids, vec![id]);
}
