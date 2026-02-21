use crate::anthropic::Message;
use dashmap::DashMap;
use std::sync::Arc;

/// Thread-safe per-user conversation history store.
pub struct ConversationStore {
    inner: Arc<DashMap<u64, Vec<Message>>>,
    /// Max number of exchange *pairs* to retain (1 pair = user + assistant message).
    max_pairs: usize,
}

impl ConversationStore {
    pub fn new(max_pairs: usize) -> Self {
        Self { inner: Arc::new(DashMap::new()), max_pairs }
    }

    pub fn add_user_message(&self, user_id: u64, content: String) {
        let mut entry = self.inner.entry(user_id).or_default();
        entry.push(Message::user(content));
        trim(&mut entry, self.max_pairs);
    }

    pub fn add_assistant_message(&self, user_id: u64, content: String) {
        let mut entry = self.inner.entry(user_id).or_default();
        entry.push(Message::assistant(content));
        trim(&mut entry, self.max_pairs);
    }

    /// Remove the last message (used to roll back a failed user message).
    pub fn pop_last(&self, user_id: u64) {
        if let Some(mut entry) = self.inner.get_mut(&user_id) {
            entry.pop();
        }
    }

    pub fn get_history(&self, user_id: u64) -> Vec<Message> {
        self.inner.get(&user_id).map(|h| h.clone()).unwrap_or_default()
    }

    pub fn clear(&self, user_id: u64) {
        self.inner.remove(&user_id);
    }
}

/// Trim history so it never exceeds `max_pairs` exchanges.
/// Always ensures the first message is from the user (Anthropic API requirement).
fn trim(history: &mut Vec<Message>, max_pairs: usize) {
    let limit = max_pairs * 2;

    // Drop from the front until we're within the limit
    while history.len() > limit {
        history.remove(0);
    }

    // The API requires the conversation to start with a user message.
    // If trimming left an orphaned assistant message at the front, drop it.
    while history.first().map(|m| m.role.as_str()) == Some("assistant") {
        history.remove(0);
    }
}
