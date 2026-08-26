//! HTTP/1.1 minimale per s-server — std::net puro, zero dipendenze.
//!
//! Scope F2: richieste con Content-Length, risposte JSON con
//! Connection: close. Un thread per connessione.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

pub enum ParseError {
    /// Richiesta malformata → 400
    Bad(String),
    /// Flusso chiuso / vuoto → chiudi silenziosamente
    Closed,
}

pub fn read_request(stream: &mut TcpStream) -> Result<Request, ParseError> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|_| ParseError::Closed)?);

    // request line
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|_| ParseError::Closed)?;
    if line.trim().is_empty() {
        return Err(ParseError::Closed);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or_else(|| ParseError::Bad("request line".into()))?.to_string();
    let path = parts.next().ok_or_else(|| ParseError::Bad("path mancante".into()))?.to_string();

    // headers
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h).map_err(|_| ParseError::Closed)?;
        if n == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = h.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length =
                    value.trim().parse().map_err(|_| ParseError::Bad("content-length".into()))?;
            }
        }
    }

    // body
    const MAX_BODY: usize = 4 * 1024 * 1024; // 4 MB basta e avanza per i prompt
    if content_length > MAX_BODY {
        return Err(ParseError::Bad(format!("body troppo grande: {content_length}")));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).map_err(|_| ParseError::Bad("body troncato".into()))?;
    }

    Ok(Request { method, path, body })
}

/// Risposta JSON completa con chiusura della connessione.
pub fn respond_json(stream: &mut TcpStream, status: u16, reason: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

// ── SSE (AC-1) ──

/// Header di apertura stream SSE: delimitazione da EOF via Connection: close.
pub fn respond_sse_open(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\r\n",
    )?;
    stream.flush()
}

/// Un evento `data: <json>\n\n` con flush immediato.
pub fn sse_data(stream: &mut TcpStream, json: &str) -> std::io::Result<()> {
    stream.write_all(b"data: ")?;
    stream.write_all(json.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

/// Evento di terminazione OpenAI-compatible.
pub fn sse_done(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()
}
