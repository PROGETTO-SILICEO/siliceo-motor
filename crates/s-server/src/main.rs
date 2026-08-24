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

mod http;

use s_config::json::{parse, write_escaped, Json};
use s_config::{FileConfig, GenerateDefaults};
use s_models::generate::{generate, GenerateParams};
use s_models::Model;
use s_tokenizer::Tokenizer;
use std::fmt::Write as _;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};

struct App {
    model: Model,
    tok: Tokenizer,
    im_end_id: Option<u32>,
    model_name: String,
    /// Default di generazione correnti (risolti: temperatura/max_tokens sempre presenti).
    /// Mutabili a runtime via POST /v1/config.
    gen_defaults: RwLock<GenerateDefaults>,
    /// Cosa riportare in GET /v1/config sulla sezione server.
    server_report: (String, u16),
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
    let t0 = std::time::Instant::now();
    let tok = Tokenizer::from_gguf(&model_path).expect("tokenizer");
    let model = Model::load(&model_path).expect("modello");
    let app = Arc::new(App {
        im_end_id: tok.token_to_id("<|im_end|>"),
        model_name: std::path::Path::new(&model_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_path.clone()),
        tok,
        model,
        gen_defaults: RwLock::new(gen0),
        server_report: (host.clone(), port),
    });
    eprintln!("s-server: modello pronto in {:.1}s", t0.elapsed().as_secs_f32());

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

    // path senza query string
    let path = req.path.split('?').next().unwrap_or("").to_string();

    match (req.method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            let mut buf = String::from("{\"status\":\"ok\",\"model\":");
            write_escaped(&mut buf, &app.model_name);
            buf.push('}');
            respond(stream, 200, &buf);
        }
        ("GET", "/v1/config") => get_config(app, stream),
        ("POST", "/v1/config") => patch_config(app, stream, &req.body),
        ("POST", "/v1/chat/completions") => chat_completions(app, stream, &req.body),
        ("POST", "/v1/completions") => completions(app, stream, &req.body),
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

// ── endpoint ──

fn completions(app: &App, stream: &mut TcpStream, body: &[u8]) {
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

    let ids = match app.tok.encode(&prompt) {
        Ok(i) => i,
        Err(e) => return error(stream, 400, e.to_string()),
    };

    let out = run_generate(app, &ids, &params);
    respond_json_completion(stream, out.finish_reason, &out.text, ids.len(), &out);
}

fn chat_completions(app: &App, stream: &mut TcpStream, body: &[u8]) {
    let j = match parse_body(body) {
        Ok(j) => j,
        Err(e) => return error(stream, 400, e),
    };
    let messages = match j.get("messages").and_then(|m| m.as_arr()) {
        Some(m) if !m.is_empty() => m,
        _ => return error(stream, 400, "campo 'messages' mancante o vuoto"),
    };
    let prompt = match render_chatml(messages) {
        Ok(p) => p,
        Err(e) => return error(stream, 400, e),
    };
    let params = match params_from_json(app, &j) {
        Ok(p) => p,
        Err(e) => return error(stream, 400, e),
    };
    // EOS di chat: <|im_end|> chiude il turno dell'assistente
    let mut params = params;
    if let Some(id) = app.im_end_id {
        params.extra_eos.push(id);
    }

    let ids = match app.tok.encode(&prompt) {
        Ok(i) => i,
        Err(e) => return error(stream, 400, e.to_string()),
    };

    let out = run_generate(app, &ids, &params);
    respond_json_chat(stream, out.finish_reason, &out.text, ids.len(), &out);
}

// ── motore ──

/// Esegue la generazione (bloccante) e ritorna il risultato.
fn run_generate<'a>(
    app: &'a App,
    ids: &[u32],
    params: &GenerateParams,
) -> s_models::generate::Generated {
    generate(&app.model, &app.tok, ids, params)
}

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

/// Template ChatML (Qwen2.5):
/// <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
fn render_chatml(messages: &[Json]) -> Result<String, String> {
    let mut prompt = String::new();
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).ok_or("message senza 'role'")?;
        let content = m.get("content").and_then(|c| c.as_str()).ok_or("message senza 'content'")?;
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
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
