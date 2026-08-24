//! s-server — server di inferenza CPU end-to-end per siliceo-motor (F2).
//!
//! Endpoint OpenAI-compatible:
//!   GET  /health
//!   POST /v1/completions        {prompt, temperature?, top_p?, top_k?, max_tokens?, stop?, seed?}
//!   POST /v1/chat/completions   {messages:[{role,content}], ...stessi parametri}
//!
//! Zero dipendenze HTTP/JSON: std::net + parser scritto in casa.
//! Uso: s-server <modello.gguf> [--port N]

mod http;
mod json;

use json::{parse, Json};
use s_models::generate::{generate, GenerateParams};
use s_models::Model;
use s_tokenizer::Tokenizer;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

struct App {
    model: Model,
    tok: Tokenizer,
    im_end_id: Option<u32>,
    model_name: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        eprintln!("uso: s-server <modello.gguf> [--port N]");
        std::process::exit(1);
    });
    let mut port: u16 = 8096;
    while let Some(a) = args.next() {
        if a == "--port" {
            port = args.next().and_then(|p| p.parse().ok()).unwrap_or(port);
        }
    }

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
    });
    eprintln!("s-server: modello pronto in {:.1}s", t0.elapsed().as_secs_f32());

    let addr = format!("127.0.0.1:{port}");
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
            json::write_escaped(&mut buf, &app.model_name);
            buf.push('}');
            respond(stream, 200, &buf);
        }
        ("POST", "/v1/chat/completions") => chat_completions(app, stream, &req.body),
        ("POST", "/v1/completions") => completions(app, stream, &req.body),
        _ => error(stream, 404, format!("endpoint sconosciuto: {} {}", req.method, req.path)),
    }
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
    let params = match params_from_json(&j) {
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
    let params = match params_from_json(&j) {
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

fn params_from_json(j: &Json) -> Result<GenerateParams, String> {
    let mut p = GenerateParams::default();
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
    json::write_escaped(&mut buf, &msg.to_string());
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
    json::write_escaped(&mut buf, &completion_id());
    buf.push_str(",\"object\":\"text_completion\",\"choices\":[{\"text\":");
    json::write_escaped(&mut buf, text);
    buf.push_str(",\"finish_reason\":");
    json::write_escaped(&mut buf, finish_reason);
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
    json::write_escaped(&mut buf, &completion_id());
    buf.push_str(",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    json::write_escaped(&mut buf, text);
    buf.push_str("},\"finish_reason\":");
    json::write_escaped(&mut buf, finish_reason);
    buf.push_str("}],");
    buf.push_str(&usage_json(prompt_tokens, out));
    buf.push('}');
    respond(stream, 200, &buf);
}
