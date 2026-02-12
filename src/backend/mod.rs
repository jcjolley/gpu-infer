//! Model backend trait and implementations.
//!
//! The backend trait is the clean cutpoint between the scheduler/pool layer
//! and the actual model inference. Any backend that implements `ModelBackend`
//! can be used with the scheduler — llama.cpp, candle, or anything else.

pub mod llama_cpp;

use crate::pool::kv_cache::PagedKvCache;
use crate::pool::{PoolError, VramPool};
use thiserror::Error;

/// Errors from model backends.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to load model: {0}")]
    LoadError(String),

    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error("VRAM pool error: {0}")]
    Pool(#[from] PoolError),
}

/// Static model configuration (read from model metadata).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of query/output attention heads.
    pub num_qo_heads: u32,
    /// Number of key/value attention heads (GQA).
    pub num_kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Maximum sequence length the model supports.
    pub max_seq_len: u32,
    /// Model name/path for display.
    pub name: String,
}

/// The backend trait — the contract between scheduler and model.
///
/// Implementations handle weight loading, tokenization, and the forward pass.
/// The scheduler provides KV cache and VRAM pool; the backend uses them
/// for attention computation.
pub trait ModelBackend {
    /// Load a model from a file path.
    ///
    /// For llama.cpp: expects a GGUF file.
    /// For candle: expects a directory with safetensors + config.json.
    fn load(path: &str, pool: &mut VramPool) -> Result<Self, BackendError>
    where
        Self: Sized;

    /// Run one decode step for a batch of sequences.
    ///
    /// Takes one token per sequence (the most recently generated token),
    /// runs the full transformer forward pass, and returns logits for
    /// each sequence's next token.
    ///
    /// `tokens`: one token ID per sequence in the batch.
    /// Returns: `[batch_size, vocab_size]` logits (flattened).
    fn decode_step(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut PagedKvCache,
        pool: &mut VramPool,
    ) -> Result<Vec<f32>, BackendError>;

    /// Tokenize text into token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, BackendError>;

    /// Detokenize token IDs back to text.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, BackendError>;

    /// Get the model's static configuration.
    fn config(&self) -> &ModelConfig;

    /// EOS token ID for this model.
    fn eos_token_id(&self) -> u32;
}
