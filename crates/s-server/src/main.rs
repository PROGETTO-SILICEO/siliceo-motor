//! s-server — server di inferenza CPU end-to-end per siliceo-motor (F2 + D6).
//!
//! Endpoint OpenAI-compatible:
//!   GET  /health
//!   POST /v1/completions        {prompt, temperature?, top_p?, top_k?, max_tokens?, stop?, seed?}
//!   POST /v1/chat/completions   {messages:[{role,content}], ...stessi parametri}
//!
//! Configurazione dinamica (D6):
//!   GET  /v1/config             config effettiva (strati fusi)
//!   POST /v1/config             patch dei default di generazione A SERVER ACCESO
//!
//! Strati (dal più debole al più forte): default < motor.json < flag CLI <
//! parametri della singola richiesta.
//!
//! Zero dipendenze HTTP/JSON: std::net + parser scritto in casa (in s-config).
//! Uso: s-server [modello.gguf] [--config PATH] [--port N] [--host H]
//!      se non c'è --config e esiste ./motor.json viene caricato da solo.

mod grammar;
mod http;
mod stream;
mod tools;

use s_config::json::{parse, write_escaped, Json};
use s_config::{FileConfig, GenerateDefaults};
use s_gguf::GgufFile;
use s_models::chat::ChatTemplate;
use s_models::generate::GenerateParams;
use s_models::Model;
use s_tokenizer::Tokenizer;
use std::fmt::Write as _;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};

/// Tutto ciò che riguarda il modello CARICATO. Sostituibile a caldo:
/// ogni richiesta prende uno snapshot (`Arc<Inner>`) e lavora su quello,
/// quindi una richiesta in corso finisce col modello su cui è iniziata.
struct Inner {
    model: Model,
    tok: Tokenizer,
    model_name: String,
    /// Template di chat del modello (D6d): rilevato dal GGUF o dalla config.
    chat: ChatTemplate,
    /// Tabella id→testo decodificato per il mascheramento grammaticale
    /// (AC-3): costruita LAZY alla prima richiesta vincolata (~150k decode).
    vocab_decoded: std::sync::Mutex<Option<std::sync::Arc<Vec<String>>>>,
}

impl Inner {
    fn vocab_table(&self) -> std::sync::Arc<Vec<String>> {
        let mut slot = self.vocab_decoded.lock().unwrap();
        if let Some(t) = &*slot {
            return std::sync::Arc::clone(t);
        }
        let mut texts = Vec::with_capacity(self.tok.vocab_size());
        for id in 0..self.tok.vocab_size() as u32 {
            texts.push(self.tok.decode(&[id]).unwrap_or_default());
        }
        let arc = std::sync::Arc::new(texts);
        *slot = Some(std::sync::Arc::clone(&arc));
        arc
    }
}

impl Inner {
    /// Carica un GGUF dal disco. Lento (~secondi): va chiamato FUORI dal
    /// lock di scrittura, così il server continua a servire col vecchio modello.
    fn load(model_path: &str, template_over: Option<&str>) -> Result<Arc<Self>, String> {
        let t0 = std::time::Instant::now();
        let tok =
            Tokenizer::from_gguf(model_path).map_err(|e| format!("tokenizer: {e}"))?;
        let model = Model::load(model_path).map_err(|e| format!("modello: {e}"))?;

        // metadata per il template di chat (riapertura leggera, solo KV)
        let gguf = GgufFile::open(model_path).map_err(|e| e.to_string())?;
        let kv_str = |key: &str| {
            gguf.metadata.iter().find(|(k, _)| k == key).and_then(|(_, v)| v.as_str())
        };
        let arch = kv_str("general.architecture").unwrap_or("sconosciuta").to_string();
        let chat = ChatTemplate::detect(
            &arch,
            kv_str("tokenizer.chat_template"),
            template_over,
        )?;
        eprintln!(
            "load: arch={arch}, template={} ({})",
            chat.format.name(),
            chat.source
        );

        let model_name = std::path::Path::new(model_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_path.to_string());
        eprintln!(
            "s-server: {model_name} pronto in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        Ok(Arc::new(Self {
            model,
            tok,
            model_name,
            chat,
            vocab_decoded: std::sync::Mutex::new(None),
        }))
    }
}

struct App {
    /// Il modello corrente, sostituibile atomicamente (POST /v1/model).
    inner: RwLock<Arc<Inner>>,
    /// Default di generazione correnti (risolti). Mutabili via POST /v1/config.
    gen_defaults: RwLock<GenerateDefaults>,
    /// Cosa riportare in GET /v1/config sulla sezione server.
    server_report: (String, u16),
    /// Override template di chat da config/CLI (None = auto), valido
    /// anche per i modelli caricati a caldo.
    template_over: Option<String>,
}

fn main() {
    // ── strato 4: riga di comando ──
    let mut args = std::env::args().skip(1);
    let mut positional_model: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut cli_port: Option<u16> = None;
    let mut cli_host: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config_path = args.next(),
            "--port" => cli_port = args.next().and_then(|p| p.parse().ok()),
            "--host" => cli_host = args.next(),
            "--model" => positional_model = args.next(),
            other => {
                if positional_model.is_none() && !other.starts_with('-') {
                    positional_model = Some(other.into());
                } else {
                    eprintln!("uso: s-server [modello.gguf] [--config PATH] [--port N] [--host H]");
                    std::process::exit(1);
                }
            }
        }
    }

    // ── strati 2+3: sistema < utente ──
    let mut file_cfg = FileConfig::default();
    let percorso = config_path.clone().or_else(|| {
        // auto-discovery: prima ./motor.json, poi il percorso di sistema
        ["motor.json", "/etc/siliceo-motor/motor.json"]
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .map(|p| p.to_string())
    });
    if let Some(p) = &percorso {
        match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| FileConfig::parse(&t)) {
            Ok(c) => {
                eprintln!("s-server: config da {p}");
                file_cfg = c;
            }
            Err(e) => {
                eprintln!("s-server: ERRORE nella config {p}: {e}");
                std::process::exit(1);
            }
        }
    }

    let model_path = positional_model.or(file_cfg.model_path.clone()).unwrap_or_else(|| {
        eprintln!("uso: s-server <modello.gguf> [--config PATH] [--port N] [--host H]");
        std::process::exit(1);
    });

    // precedenza: CLI > file > default
    let port = cli_port.or(file_cfg.server.port).unwrap_or(8096);
    let host =
        cli_host.or(file_cfg.server.host.clone()).unwrap_or_else(|| "127.0.0.1".into());

    // default di generazione risolti: assoluti < file (la CLI non tocca la generazione)
    let gen0 = GenerateDefaults::default().merge(&file_cfg.generate);

    eprintln!("s-server: carico {model_path} ...");
    let inner = Inner::load(&model_path, file_cfg.model_template.as_deref()).unwrap_or_else(|e| {
        eprintln!("s-server: ERRORE nel caricamento: {e}");
        std::process::exit(1);
    });
    let app = Arc::new(App {
        inner: RwLock::new(inner),
        gen_defaults: RwLock::new(gen0),
        server_report: (host.clone(), port),
        template_over: file_cfg.model_template.clone(),
    });

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr).expect("bind");
    eprintln!("s-server: in ascolto su http://{addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let app = Arc::clone(&app);
                std::thread::spawn(move || handle(&app, &mut stream));
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

fn handle(app: &App, stream: &mut TcpStream) {
    let req = match http::read_request(stream) {
        Ok(r) => r,
        Err(http::ParseError::Closed) => return,
        Err(http::ParseError::Bad(e)) => {
            return error(stream, 400, e);
        }
    };

    // snapshot del modello corrente: le richieste in corso non vedono gli swap
    let inner = Arc::clone(&app.inner.read().unwrap());

    // path senza query string
    let path = req.path.split('?').next().unwrap_or("").to_string();

    match (req.method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            let mut buf = String::from("{\"status\":\"ok\",\"model\":");
            write_escaped(&mut buf, &inner.model_name);
            let _ = write!(buf, ",\"layers\":{},\"embd\":{},\"chat\":\"{}\",\"tpl_from\":\"{}\"}}",
                inner.model.config.n_layers, inner.model.config.n_embd,
                inner.chat.format.name(), inner.chat.source);
            respond(stream, 200, &buf);
        }
        ("GET", "/v1/config") => get_config(app, stream),
        ("POST", "/v1/config") => patch_config(app, stream, &req.body),
        ("POST", "/v1/model") => load_model(app, stream, &req.body),
        ("POST", "/v1/chat/completions") => chat_completions(&inner, app, stream, &req.body),
        ("POST", "/v1/completions") => completions(&inner, app, stream, &req.body),
        _ => error(stream, 404, format!("endpoint sconosciuto: {} {}", req.method, req.path)),
    }
}

// ── configurazione dinamica (D6) ──

fn get_config(app: &App, stream: &mut TcpStream) {
    let g = app.gen_defaults.read().unwrap();
    let mut buf = String::from("{\"server\":{");
    buf.push_str("\"host\":");
    write_escaped(&mut buf, &app.server_report.0);
    let _ = write!(buf, ",\"port\":{}}},\"generate\":{{", app.server_report.1);
    let _ = write!(buf, "\"temperature\":{}", g.temperature.unwrap_or(0.0));
    match g.top_p {
        Some(v) => { let _ = write!(buf, ",\"top_p\":{v}"); }
        None => buf.push_str(",\"top_p\":null"),
    }
    match g.top_k {
        Some(v) => { let _ = write!(buf, ",\"top_k\":{v}"); }
        None => buf.push_str(",\"top_k\":null"),
    }
    let _ = write!(buf, ",\"max_tokens\":{}}}", g.max_tokens.unwrap_or(256));
    buf.push('}');
    drop(g);
    respond(stream, 200, &buf);
}

/// Patch dei default di generazione a server acceso.
/// Accetta sia `{"temperature":0.7}` sia `{"generate":{"temperature":0.7}}`.
fn patch_config(app: &App, stream: &mut TcpStream, body: &[u8]) {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(_) => return error(stream, 400, "body non è UTF-8"),
    };
    // accetta anche la forma annidata {"generate":{...}}
    let effettivo = match parse(text)
        .map_err(|e| format!("JSON non valido: {e}"))
        .and_then(|j| {
            if j.get("generate").is_some() {
                FileConfig::from_json(&j).map(|c| c.generate)
            } else {
                GenerateDefaults::patch_from_str(text)
            }
        }) {
        Ok(patch) => patch,
        Err(e) => return error(stream, 400, e),
    };

    let nuovo = {
        let correnti = app.gen_defaults.read().unwrap();
        correnti.merge(&effettivo)
    };
    *app.gen_defaults.write().unwrap() = nuovo.clone();
    let _ = nuovo; // la risposta la costruiamo rileggendo (semplice e senza lock doppio)
    get_config(app, stream);
}

/// Hot-swap del modello (D6c): `{"path":"altro.gguf"}`.
/// Il caricamento avviene FUORI dal lock: finché non riesce, il server
/// continua a servire col modello vecchio. Se fallisce → errore pulito,
/// nessun stato toccato.
fn load_model(app: &App, stream: &mut TcpStream, body: &[u8]) {
    let j = match parse_body(body) {
        Ok(j) => j,
        Err(e) => return error(stream, 400, e),
    };
    let nuovo_path = match j.get("path").and_then(|p| p.as_str()) {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => return error(stream, 400, "campo 'path' mancante o vuoto"),
    };
    if !std::path::Path::new(&nuovo_path).exists() {
        return error(stream, 400, format!("file non trovato: {nuovo_path}"));
    }

    eprintln!("s-server: hot-swap verso {nuovo_path} ...");
    let nuovo = match Inner::load(&nuovo_path, app.template_over.as_deref()) {
        Ok(n) => n,
        Err(e) => return error(stream, 400, format!("caricamento fallito ({e}): il modello precedente resta attivo")),
    };

    let nome = nuovo.model_name.clone();
    *app.inner.write().unwrap() = nuovo;

    let mut buf = String::from("{\"status\":\"ok\",\"model\":");
    write_escaped(&mut buf, &nome);
    buf.push('}');
    respond(stream, 200, &buf);
}

// ── endpoint ──

fn completions(inner: &Inner, app: &App, stream: &mut TcpStream, body: &[u8]) {
    let j = match parse_body(body) {
        Ok(j) => j,
        Err(e) => return error(stream, 400, e),
    };
    let prompt = match j.get("prompt").and_then(|p| p.as_str()) {
        Some(p) => p.to_string(),
        None => return error(stream, 400, "campo 'prompt' mancante o non stringa"),
    };
    let params = match params_from_json(app, &j) {
        Ok(p) => p,
        Err(e) => return error(stream, 400, e),
    };

    let ids = match inner.tok.encode(&prompt) {
        Ok(i) => i,
        Err(e) => return error(stream, 400, e.to_string()),
    };

    let vuole_stream = j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    if vuole_stream {
        // SSE: dopo [DONE] la connessione si chiude — nessuna seconda risposta
        run_streaming_v(inner, &ids, &params, stream, false, vuole_json(&j));
        return;
    }

    let out = run_generate_con_vincolo(inner, &ids, &params, vuole_json(&j), &mut |_| true);
    respond_json_completion(stream, out.finish_reason, &out.text, ids.len(), &out);
}

fn chat_completions(inner: &Inner, app: &App, stream: &mut TcpStream, body: &[u8]) {
    let j = match parse_body(body) {
        Ok(j) => j,
        Err(e) => return error(stream, 400, e),
    };
    let messages = match j.get("messages").and_then(|m| m.as_arr()) {
        Some(m) if !m.is_empty() => m,
        _ => return error(stream, 400, "campo 'messages' mancante o vuoto"),
    };

    // ── AC-2: strumenti dichiarati dal client ──
    let defs = match tools::parse_tools(j.get("tools")) {
        Ok(t) => t,
        Err(e) => return error(stream, 400, format!("tools: {e}")),
    };
    let usa_tools = !defs.is_empty();

    // ── messaggi → coppie (role, content) per il template ──
    // assistant con tool_calls → content ricostruito con i tag <tool_call>;
    // role "tool" → passa tale quale (convenzione hermes/Qwen2.5).
    let mut coppie: Vec<(String, String)> = Vec::with_capacity(messages.len() + 1);
    if usa_tools {
        coppie.push(("system".into(), tools::tools_section(&defs)));
    }
    if vuole_json(&j) {
        // suggerimento lato prompt: la VALIDITÀ la garantisce la maschera,
        // questo aiuta solo il modello a voler chiudere presto
        coppie.push(("system".into(), "Respond with only a valid JSON object.".into()));
    }
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role.is_empty() {
            return error(stream, 400, "message senza 'role' stringa");
        }
        let mut content =
            m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();

        if role == "assistant" {
            // le chiamate fatte dal modello tornano nel prompt come le aveva
            // scritte: è ciò che i modelli hermes-style si aspettano
            if let Some(calls) = m.get("tool_calls").and_then(|c| c.as_arr()) {
                for call in calls {
                    let name = call.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                    let args = call.get("function").and_then(|f| f.get("arguments"));
                    if name.is_empty() {
                        continue;
                    }
                    let args_json = match args {
                        Some(a @ Json::Obj(_)) | Some(a @ Json::Arr(_)) => a.dump(),
                        Some(Json::Str(st)) => st.clone(),
                        _ => "{}".to_string(),
                    };
                    content.push_str("\n<tool_call>\n{\"name\": ");
                    write_json_str(&mut content, name);
                    content.push_str(", \"arguments\": ");
                    content.push_str(&args_json);
                    content.push_str("}\n</tool_call>");
                }
            }
        }

        if role == "tool" && content.is_empty() {
            return error(stream, 400, "message role 'tool' senza content");
        }
        coppie.push((role.to_string(), content));
    }
    let riferimenti: Vec<(&str, &str)> =
        coppie.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
    let prompt = inner.chat.render(&riferimenti);

    let mut params = match params_from_json(app, &j) {
        Ok(p) => p,
        Err(e) => return error(stream, 400, e),
    };
    for stop in inner.chat.format.stop_tokens() {
        if let Some(id) = inner.tok.token_to_id(stop) {
            params.extra_eos.push(id);
        }
    }

    let ids = match inner.tok.encode(&prompt) {
        Ok(i) => i,
        Err(e) => return error(stream, 400, e.to_string()),
    };

    let vuole_stream = j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // ── non-streaming: con tools attivi possiamo rilevare le chiamate ──
    if !vuole_stream {
        let json_mode = vuole_json(&j);
        let out = run_generate_con_vincolo(inner, &ids, &params, json_mode, &mut |_| true);
        let scan = if usa_tools { tools::scan(&out.text) } else { tools::scan("") };
        if usa_tools && scan.has_calls() {
            return respond_json_chat_tool_calls(
                stream,
                &scan.content,
                &scan.calls,
                ids.len(),
                &out,
            );
        }
        respond_json_chat(stream, out.finish_reason, &out.text, ids.len(), &out);
        return;
    }

    // ── streaming: il testo passa com'è (i tag <tool_call> inclusi se il
    // modello li emette; la rilevazione strutturata resta non-streaming).
    // Documentato in docs/AGENTIC_CORE.md come limite v1 di AC-2. ──
    run_streaming_v(inner, &ids, &params, stream, true, vuole_json(&j));
}

// ── streaming SSE (AC-1) ──

/// Un chunk SSE in formato OpenAI. `body_choice` è il contenuto interno
/// dell'oggetto choice GIÀ COMPLETO (delta/text + finish_reason).
fn chunk_openai(cid: &str, object: &str, body_choice: &str) -> String {
    let mut buf = String::new();
    buf.push_str("{\"id\":");
    write_escaped(&mut buf, cid);
    buf.push_str(",\"object\":");
    write_escaped(&mut buf, object);
    buf.push_str(",\"choices\":[{");
    buf.push_str(body_choice);
    buf.push_str("}]}");
    buf
}

/// Generazione con invio dei pezzi man mano. Ritorna il Generated completo
/// oppure None se la connessione SSE è caduta durante l'invio.
///
/// Il pezzo emesso è il DIFF del decode dell'intera sequenza (sempre UTF-8
/// valido), filtrato dallo StreamSlicer (trattiene la coda con stop attive).
#[allow(clippy::too_many_arguments)]
fn run_streaming_v(
    inner: &Inner,
    ids: &[u32],
    params: &GenerateParams,
    tcp: &mut TcpStream,
    is_chat: bool,
    vincolo: bool,
) -> Option<s_models::generate::Generated> {
    use http::{respond_sse_open, sse_data, sse_done};
    use stream::StreamSlicer;

    if respond_sse_open(tcp).is_err() {
        return None;
    }

    let cid = completion_id();
    let object = if is_chat { "chat.completion.chunk" } else { "text_completion.chunk" };

    // primo chunk: ruolo (convenzione OpenAI per chat; delta vuoto per completions)
    let primo = if is_chat {
        chunk_openai(&cid, object, "\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null")
    } else {
        chunk_openai(&cid, object, "\"text\":\"\",\"finish_reason\":null")
    };
    if sse_data(tcp, &primo).is_err() {
        return None;
    }

    let eos_all: Vec<u32> =
        inner.tok.eos_id().into_iter().chain(params.extra_eos.iter().copied()).collect();
    let slicer = std::cell::RefCell::new(StreamSlicer::new(&params.stop));
    let generati: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
    let mut caduto = false;

    let tok = &inner.tok;
    let mut on_token = |t: u32| -> bool {
        generati.borrow_mut().push(t);
        let full = tok.decode(&generati.borrow()).unwrap_or_default();
        let piece = slicer.borrow_mut().push(&full);
        if !piece.is_empty() {
            let mut body = String::new();
            if is_chat {
                body.push_str("\"index\":0,\"delta\":{\"content\":");
                write_escaped(&mut body, &piece);
                body.push_str("},");
            } else {
                body.push_str("\"text\":");
                write_escaped(&mut body, &piece);
                body.push(',');
            }
            body.push_str("\"finish_reason\":null");
            let chunk = chunk_openai(&cid, object, &body);
            if sse_data(tcp, &chunk).is_err() {
                caduto = true;
                return false; // client andato: ferma la generazione
            }
        }
        true
    };

    let decode = |g: &[u32]| inner.tok.decode(g).unwrap_or_default();
    let esiti = std::cell::RefCell::new(if vincolo {
        Some(grammar::JsonMasker::from_texts(inner.vocab_table()))
    } else {
        None
    });
    let mut masker = || -> Option<Vec<bool>> {
        match &*esiti.borrow() {
            Some(_) => Some(esiti.borrow().as_ref().unwrap().mask()),
            None => None,
        }
    };
    let mut on_tok = |t: u32| -> bool {
        if esiti.borrow().is_some() {
            esiti.borrow_mut().as_mut().unwrap().advance(t);
        }
        on_token(t)
    };
    let out =
        s_models::generate::generate_tokens(&inner.model, ids, params, &eos_all, &decode, &mut on_tok, &mut masker);

    if caduto {
        return None;
    }

    // coda trattenuta dalle stop-strings, poi chunk finale con finish_reason.
    // ATTENZIONE: si usa out.text (GIÀ TRONCATO alle stop dal loop), mai il
    // decode integrale degli id — altrimenti si emette ciò che la stop voleva
    // nascondere (bug trovato dal test live con stop=["capitale"]).
    let full = &out.text;
    let coda = slicer.borrow_mut().finish(full);
    if !coda.is_empty() {
        let mut body = String::new();
        if is_chat {
            body.push_str("\"index\":0,\"delta\":{\"content\":");
            write_escaped(&mut body, &coda);
            body.push_str("},");
        } else {
            body.push_str("\"text\":");
            write_escaped(&mut body, &coda);
            body.push(',');
        }
        body.push_str("\"finish_reason\":null");
        let _ = sse_data(tcp, &chunk_openai(&cid, object, &body));
    }
    let finale = chunk_openai(
        &cid,
        object,
        &format!("\"index\":0,\"delta\":{{}},\"finish_reason\":{}", escape_json(out.finish_reason)),
    );
    let _ = sse_data(tcp, &finale);
    let _ = sse_done(tcp);
    Some(out)
}

/// Stringa JSON già quotata per valori noti e sicuri ("stop"/"length").
fn escape_json(s: &str) -> String {
    let mut buf = String::new();
    write_escaped(&mut buf, s);
    buf
}


// ── AC-3: structured output ──

/// true se la richiesta chiede output JSON vincolato (response_format
/// json_object | json_schema; per lo schema-specifico vedi docs/AGENTIC_CORE.md).
fn vuole_json(j: &Json) -> bool {
    j.get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        .map(|t| t == "json_object" || t == "json_schema")
        .unwrap_or(false)
}

/// Genera con eventuale vincolo grammaticale JSON. Con `None` è la strada
/// storica (generate). Il vincolo avanza DENTRO la callback del token:
/// un solo percorso di generazione per tutte le combinazioni.
#[allow(clippy::too_many_arguments)]
fn run_generate_con_vincolo(
    inner: &Inner,
    ids: &[u32],
    params: &GenerateParams,
    vincolo: bool,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> s_models::generate::Generated {
    let eos_all: Vec<u32> =
        inner.tok.eos_id().into_iter().chain(params.extra_eos.iter().copied()).collect();
    let decode = |g: &[u32]| inner.tok.decode(g).unwrap_or_default();

    eprintln!("[ac3] vincolo json attivo: {vincolo}");
    if !vincolo {
        let mut nomask = || -> Option<Vec<bool>> { None };
        return s_models::generate::generate_tokens(
            &inner.model, ids, params, &eos_all, &decode, on_token, &mut nomask,
        );
    }

    let texts = inner.vocab_table();
    let m = std::cell::RefCell::new(grammar::JsonMasker::from_texts(texts));
    let mut on_tok = |t: u32| -> bool {
        m.borrow_mut().advance(t);
        on_token(t)
    };
    let mut mk = || -> Option<Vec<bool>> { Some(m.borrow().mask()) };
    s_models::generate::generate_tokens(
        &inner.model, ids, params, &eos_all, &decode, &mut on_tok, &mut mk,
    )
}

// ── motore ──

// ── helpers ──

fn parse_body(body: &[u8]) -> Result<Json, String> {
    let text = std::str::from_utf8(body).map_err(|_| "body non è UTF-8".to_string())?;
    parse(text)
}

fn params_from_json(app: &App, j: &Json) -> Result<GenerateParams, String> {
    // strato 5: la richiesta parte dai default correnti (strato 6, mutabili via /v1/config)
    let base = app.gen_defaults.read().unwrap().clone();
    let mut p = GenerateParams {
        max_tokens: base.max_tokens.unwrap_or(256),
        temperature: base.temperature.unwrap_or(0.0),
        top_k: base.top_k,
        top_p: base.top_p,
        seed: None,
        stop: Vec::new(),
        extra_eos: Vec::new(),
    };
    if let Some(v) = j.get("temperature") {
        p.temperature = v.as_f64().ok_or("temperature non numerica")? as f32;
    }
    if let Some(v) = j.get("top_p") {
        p.top_p = Some(v.as_f64().ok_or("top_p non numerico")? as f32);
    }
    if let Some(v) = j.get("top_k") {
        p.top_k = Some(v.as_usize().ok_or("top_k non intero")?);
    }
    if let Some(v) = j.get("max_tokens") {
        p.max_tokens = v.as_usize().ok_or("max_tokens non intero")?;
    }
    if let Some(v) = j.get("seed") {
        p.seed = Some(v.as_f64().ok_or("seed non numerico")? as u64);
    }
    if let Some(v) = j.get("stop") {
        match v {
            Json::Str(s) => p.stop.push(s.clone()),
            Json::Arr(items) => {
                for item in items {
                    p.stop.push(item.as_str().ok_or("stop: elementi non stringa")?.to_string());
                }
            }
            _ => return Err("stop deve essere stringa o array di stringhe".into()),
        }
    }
    Ok(p)
}

fn respond(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "?",
    };
    http::respond_json(stream, status, reason, body);
}

fn error(stream: &mut TcpStream, status: u16, msg: impl std::fmt::Display) {
    let mut buf = String::from("{\"error\":");
    write_escaped(&mut buf, &msg.to_string());
    buf.push('}');
    respond(stream, status, &buf);
}

fn usage_json(prompt_tokens: usize, out: &s_models::generate::Generated) -> String {
    format!(
        r#""usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}},"tokens_per_second":{:.2}"#,
        prompt_tokens,
        out.ids.len(),
        prompt_tokens + out.ids.len(),
        out.tokens_per_sec()
    )
}

/// Scrive una stringa JSON QUOTATA (virgolette incluse).
pub(crate) fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    write_escaped(out, s);
    out.push('"');
}

fn respond_json_chat_tool_calls(
    stream: &mut TcpStream,
    content: &str,
    calls: &[tools::DetectedCall],
    prompt_tokens: usize,
    out: &s_models::generate::Generated,
) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nano = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut buf = String::new();
    buf.push_str("{\"id\":");
    write_escaped(&mut buf, &completion_id());
    buf.push_str(",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    write_escaped(&mut buf, content);
    buf.push_str(",\"tool_calls\":[");
    for (i, c) in calls.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        let _ = write!(buf, "{{\"id\":\"call_{nano:x}{i}\",\"type\":\"function\",\"function\":{{\"name\":");
        write_escaped(&mut buf, &c.name);
        buf.push_str(",\"arguments\":");
        write_escaped(&mut buf, &c.arguments);
        buf.push_str("}}");
    }
    buf.push_str("]},\"finish_reason\":\"tool_calls\"}],");
    buf.push_str(&usage_json(prompt_tokens, out));
    buf.push('}');
    respond(stream, 200, &buf);
}

fn completion_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("cmpl-{n:x}")
}

fn respond_json_completion(
    stream: &mut TcpStream,
    finish_reason: &str,
    text: &str,
    prompt_tokens: usize,
    out: &s_models::generate::Generated,
) {
    let mut buf = String::new();
    buf.push_str("{\"id\":");
    write_escaped(&mut buf, &completion_id());
    buf.push_str(",\"object\":\"text_completion\",\"choices\":[{\"text\":");
    write_escaped(&mut buf, text);
    buf.push_str(",\"finish_reason\":");
    write_escaped(&mut buf, finish_reason);
    buf.push_str("}],");
    buf.push_str(&usage_json(prompt_tokens, out));
    buf.push('}');
    respond(stream, 200, &buf);
}

fn respond_json_chat(
    stream: &mut TcpStream,
    finish_reason: &str,
    text: &str,
    prompt_tokens: usize,
    out: &s_models::generate::Generated,
) {
    let mut buf = String::new();
    buf.push_str("{\"id\":");
    write_escaped(&mut buf, &completion_id());
    buf.push_str(",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    write_escaped(&mut buf, text);
    buf.push_str("},\"finish_reason\":");
    write_escaped(&mut buf, finish_reason);
    buf.push_str("}],");
    buf.push_str(&usage_json(prompt_tokens, out));
    buf.push('}');
    respond(stream, 200, &buf);
}
