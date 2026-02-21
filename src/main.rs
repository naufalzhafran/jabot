mod anthropic;
mod config;
mod conversation;

use anthropic::AnthropicClient;
use config::Config;
use conversation::ConversationStore;

use std::sync::Arc;

use teloxide::{prelude::*, types::ChatAction, utils::command::BotCommands};

// ── Shared bot state ──────────────────────────────────────────────────────────

struct BotState {
    config: Config,
    anthropic: AnthropicClient,
    conversations: ConversationStore,
}

// ── Commands ──────────────────────────────────────────────────────────────────

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Available commands:")]
enum Command {
    #[command(description = "Start a conversation with Jabot")]
    Start,
    #[command(description = "Clear your conversation history")]
    Clear,
    #[command(description = "Show this help message")]
    Help,
}

// ── Entry point ───────────────────────────────────────────────────────────────

type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    pretty_env_logger::init();

    log::info!("Starting Jabot…");

    let config = Config::load("config.toml").expect("❌  Failed to load config.toml");

    let anthropic_key = std::env::var("MINIMAX_API_KEY")
        .expect("❌  MINIMAX_API_KEY is not set — add it to your .env file");

    let telegram_token = std::env::var("TELEGRAM_BOT_TOKEN")
        .expect("❌  TELEGRAM_BOT_TOKEN is not set — add it to your .env file");

    let state = Arc::new(BotState {
        conversations: ConversationStore::new(config.model.max_history),
        anthropic: AnthropicClient::new(anthropic_key),
        config,
    });

    let bot = Bot::new(telegram_token);

    log::info!("Bot ready — dispatching messages");

    // Route: commands first, then free-form text
    let handler = Update::filter_message()
        .branch(
            Message::filter_text()
                .filter_command::<Command>()
                .endpoint(handle_command),
        )
        .branch(Message::filter_text().endpoint(handle_message));

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

// ── Command handler ───────────────────────────────────────────────────────────

async fn handle_command(
    bot: Bot,
    msg: Message,
    cmd: Command,
    state: Arc<BotState>,
) -> HandlerResult {
    match cmd {
        Command::Start => {
            let name = &state.config.personality.name;
            bot.send_message(
                msg.chat.id,
                format!("Hey! I'm {name}, your personal AI assistant. What's on your mind?"),
            )
            .await?;
        }

        Command::Clear => {
            if let Some(user) = msg.from.as_ref() {
                state.conversations.clear(user.id.0);
                log::info!("Cleared history for user {}", user.id.0);
            }
            bot.send_message(msg.chat.id, "Memory cleared — fresh slate!")
                .await?;
        }

        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string())
                .await?;
        }
    }
    Ok(())
}

// ── Message handler ───────────────────────────────────────────────────────────

async fn handle_message(bot: Bot, msg: Message, state: Arc<BotState>) -> HandlerResult {
    // Only process plain text; skip unknown commands (start with '/')
    let text = match msg.text() {
        Some(t) if !t.starts_with('/') => t.to_string(),
        _ => return Ok(()),
    };

    let user_id = match msg.from.as_ref() {
        Some(u) => u.id.0,
        None => return Ok(()),
    };

    log::info!("Message from {user_id}: {text}");

    // Show typing indicator while we wait for the API
    bot.send_chat_action(msg.chat.id, ChatAction::Typing)
        .await?;

    // Store user message and build the full history to send
    state.conversations.add_user_message(user_id, text.clone());
    let history = state.conversations.get_history(user_id);

    // Call Claude
    let result = state
        .anthropic
        .chat(
            &state.config.model.id,
            state.config.model.max_tokens,
            &state.config.personality.system_prompt,
            history,
            state.config.personality.temperature,
        )
        .await;

    match result {
        Ok(reply) => {
            state
                .conversations
                .add_assistant_message(user_id, reply.clone());
            log::info!("Reply to {user_id}: {reply}");

            // Telegram messages are capped at 4096 chars — split if needed
            for chunk in split_message(&reply) {
                bot.send_message(msg.chat.id, chunk).await?;
            }
        }

        Err(e) => {
            log::error!("API error for user {user_id}: {e}");
            // Roll back the user message so next attempt has a clean history
            state.conversations.pop_last(user_id);
            bot.send_message(
                msg.chat.id,
                "Hmm, something went sideways on my end. Give it another shot!",
            )
            .await?;
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split a response into ≤4096-byte chunks, preferring newline boundaries.
fn split_message(text: &str) -> Vec<String> {
    const MAX: usize = 4096;

    if text.len() <= MAX {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + MAX).min(text.len());

        // Try to break at the last newline within the window
        let split_at = if end < text.len() {
            text[start..end]
                .rfind('\n')
                .map(|i| start + i + 1)
                .unwrap_or(end)
        } else {
            end
        };

        chunks.push(text[start..split_at].to_string());
        start = split_at;
    }

    chunks
}
