use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub personality: PersonalityConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PersonalityConfig {
    pub name: String,
    pub system_prompt: String,
    /// Temperature for Claude responses (0.0–1.0). None uses API default.
    pub temperature: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub max_tokens: u32,
    /// Max conversation pairs to keep in memory (1 pair = user msg + bot reply)
    pub max_history: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Could not read config file: {path}"))?;
        toml::from_str(&content).with_context(|| format!("Failed to parse {path}"))
    }
}
