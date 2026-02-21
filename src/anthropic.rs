use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

const API_URL: &str = "https://api.minimax.io/anthropic/v1/messages";
const API_VERSION: &str = "2023-06-01";

// ── Public message type (shared with conversation store) ──────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

// ── Internal request / response types ─────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Response {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// ── Client ────────────────────────────────────────────────────────────────────

pub struct AnthropicClient {
    http: Client,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: Client::new(),
            api_key,
        }
    }

    /// Send a full conversation to Claude and return the text reply.
    pub async fn chat(
        &self,
        model: &str,
        max_tokens: u32,
        system: &str,
        messages: Vec<Message>,
        temperature: Option<f64>,
    ) -> Result<String> {
        let body = Request {
            model,
            max_tokens,
            system,
            messages,
            temperature,
        };

        // Log the payload being sent to MiniMax
        if let Ok(json) = serde_json::to_string_pretty(&body) {
            log::info!("Sending payload to MiniMax:\n{}", json);
        }

        let http_resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Network error contacting MiniMax API")?;

        let status = http_resp.status();
        let raw = http_resp
            .text()
            .await
            .context("Failed to read MiniMax response body")?;

        if !status.is_success() {
            // Try to surface the structured error message
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&raw) {
                anyhow::bail!("MiniMax {} — {}", err.error.error_type, err.error.message);
            }
            anyhow::bail!("MiniMax API error {status}: {raw}");
        }

        let resp: Response =
            serde_json::from_str(&raw).context("Failed to parse MiniMax response JSON")?;

        // Collect all text blocks (ignore thinking blocks, etc.)
        let text = resp
            .content
            .into_iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n");

        if text.is_empty() {
            anyhow::bail!("MiniMax returned an empty response");
        }

        Ok(text)
    }
}
