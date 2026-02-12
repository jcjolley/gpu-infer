//! llama.cpp backend — GGUF model loading and inference via llama-cpp-2.
//!
//! This backend delegates inference entirely to llama.cpp. It handles:
//! - GGUF model loading (any architecture llama.cpp supports)
//! - Tokenization and detokenization
//! - KV cache management (llama.cpp's own implementation)
//! - CUDA acceleration (llama.cpp manages its own GPU context)
//!
//! Our scheduler handles the coordination layer above: multi-sequence
//! lifecycle management and the self-feeding loop.

#[cfg(feature = "llama-cpp")]
mod inner {
    use crate::backend::{BackendError, ModelConfig};

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::LlamaToken;

    use std::path::Path;

    /// llama.cpp backend wrapping the llama-cpp-2 crate.
    ///
    /// Field ordering matters: ctx drops before model drops before backend.
    /// The context borrows from model, so model must outlive context.
    pub struct LlamaCppBackend {
        // IMPORTANT: ctx must be declared BEFORE model so it drops first.
        // Rust drops fields in declaration order.
        ctx: LlamaContext<'static>,
        model: Box<LlamaModel>,
        _backend: Box<LlamaBackend>,
        config: ModelConfig,
    }

    impl LlamaCppBackend {
        /// Load a GGUF model.
        ///
        /// `path`: Path to the .gguf file.
        /// `n_gpu_layers`: How many layers to offload to GPU.
        ///   Use `u32::MAX` for full GPU, `0` for CPU only,
        ///   or a specific number for partial offload.
        /// `ctx_size`: Context window size (max sequence length).
        /// `n_seq_max`: Maximum concurrent sequences (default 1 if you only
        ///   need single-sequence generation).
        pub fn load_gguf(
            path: &str,
            n_gpu_layers: u32,
            ctx_size: u32,
            n_seq_max: u32,
        ) -> Result<Self, BackendError> {
            if !Path::new(path).exists() {
                return Err(BackendError::LoadError(format!(
                    "GGUF file not found: {}",
                    path
                )));
            }

            let backend = Box::new(
                LlamaBackend::init()
                    .map_err(|e| BackendError::LoadError(format!("backend init: {}", e)))?,
            );

            let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

            let model = Box::new(
                LlamaModel::load_from_file(&backend, path, &model_params)
                    .map_err(|e| BackendError::LoadError(format!("{}", e)))?,
            );

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(ctx_size))
                .with_n_batch(512)
                .with_n_seq_max(n_seq_max);

            // SAFETY: model is in a Box (heap-allocated, stable address).
            // We transmute the lifetime to 'static because the Box<LlamaModel>
            // is stored in the same struct and outlives the context (ctx drops first
            // due to field declaration order).
            let ctx = {
                let model_ref: &'static LlamaModel =
                    unsafe { &*(&*model as *const LlamaModel) };
                model_ref
                    .new_context(&backend, ctx_params)
                    .map_err(|e| BackendError::LoadError(format!("context: {}", e)))?
            };

            let n_embd = model.n_embd() as u32;
            let n_head = model.n_head();
            let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };

            let config = ModelConfig {
                num_qo_heads: n_head,
                num_kv_heads: model.n_head_kv(),
                head_dim,
                num_layers: model.n_layer(),
                vocab_size: model.n_vocab() as u32,
                max_seq_len: ctx_size,
                name: Path::new(path)
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.to_string()),
            };

            Ok(Self {
                ctx,
                model,
                _backend: backend,
                config,
            })
        }

        /// Get the model's EOS token.
        pub fn eos_token(&self) -> u32 {
            self.model.token_eos().0 as u32
        }

        /// Tokenize a string, optionally adding BOS.
        pub fn tokenize_str(
            &self,
            text: &str,
            add_bos: bool,
        ) -> Result<Vec<u32>, BackendError> {
            let bos = if add_bos { AddBos::Always } else { AddBos::Never };
            let tokens = self
                .model
                .str_to_token(text, bos)
                .map_err(|e| BackendError::TokenizerError(format!("{}", e)))?;
            Ok(tokens.iter().map(|t| t.0 as u32).collect())
        }

        /// Detokenize tokens back to a string.
        pub fn detokenize_tokens(&self, tokens: &[u32]) -> Result<String, BackendError> {
            let mut result = Vec::new();
            for &tok in tokens {
                let bytes = self
                    .model
                    .token_to_piece_bytes(LlamaToken(tok as i32), 128, false, None)
                    .map_err(|e| BackendError::TokenizerError(format!("{}", e)))?;
                result.extend_from_slice(&bytes);
            }
            String::from_utf8(result)
                .map_err(|e| BackendError::TokenizerError(format!("utf8: {}", e)))
        }

        /// Run a forward pass: feed tokens, get logits for the last position.
        ///
        /// Handles both prefill (multiple tokens) and single-token decode.
        pub fn forward(
            &mut self,
            tokens: &[u32],
            seq_id: i32,
            pos_start: i32,
        ) -> Result<Vec<f32>, BackendError> {
            let n_tokens = tokens.len();
            let mut batch = LlamaBatch::new(n_tokens, 1);

            for (i, &tok) in tokens.iter().enumerate() {
                let is_last = i == n_tokens - 1;
                batch
                    .add(
                        LlamaToken(tok as i32),
                        pos_start + i as i32,
                        &[seq_id],
                        is_last,
                    )
                    .map_err(|e| BackendError::InferenceError(format!("batch add: {}", e)))?;
            }

            self.ctx
                .decode(&mut batch)
                .map_err(|e| BackendError::InferenceError(format!("decode: {}", e)))?;

            // Get logits for the last token position
            let logits = self.ctx.get_logits_ith((n_tokens - 1) as i32);
            Ok(logits.to_vec())
        }

        /// Batched forward pass: feed tokens from multiple sequences in one decode call.
        ///
        /// Each entry is `(token, seq_id, position)`. Logits are requested at each
        /// entry's position so you can sample per-sequence afterwards.
        ///
        /// Batch indices are sequential starting at 0 — the i-th entry's logits
        /// are at batch index `i`. Use `sample_next(sampler, i as i32)` after this.
        pub fn forward_batch(
            &mut self,
            entries: &[(u32, i32, i32)],
        ) -> Result<(), BackendError> {
            if entries.is_empty() {
                return Ok(());
            }

            let mut batch = LlamaBatch::new(entries.len(), 1);

            for &(token, seq_id, pos) in entries {
                batch
                    .add(LlamaToken(token as i32), pos, &[seq_id], true)
                    .map_err(|e| BackendError::InferenceError(format!("batch add: {}", e)))?;
            }

            self.ctx
                .decode(&mut batch)
                .map_err(|e| BackendError::InferenceError(format!("decode: {}", e)))?;

            Ok(())
        }

        /// Batched prefill: feed prompt tokens from multiple sequences in one decode call.
        ///
        /// Each entry is `(tokens, seq_id)`. Position starts at 0 for each sequence.
        /// Logits are requested only at the last token of each sequence (for sampling).
        ///
        /// Returns the batch indices of each sequence's last token, in input order.
        pub fn forward_prefill_batch(
            &mut self,
            sequences: &[(&[u32], i32)],
        ) -> Result<Vec<i32>, BackendError> {
            if sequences.is_empty() {
                return Ok(Vec::new());
            }

            let total_tokens: usize = sequences.iter().map(|(toks, _)| toks.len()).sum();
            let mut batch = LlamaBatch::new(total_tokens, 1);
            let mut last_indices = Vec::with_capacity(sequences.len());
            let mut batch_idx = 0i32;

            for &(tokens, seq_id) in sequences {
                let n = tokens.len();
                for (i, &tok) in tokens.iter().enumerate() {
                    let is_last = i == n - 1;
                    batch
                        .add(LlamaToken(tok as i32), i as i32, &[seq_id], is_last)
                        .map_err(|e| {
                            BackendError::InferenceError(format!("batch add: {}", e))
                        })?;
                    if is_last {
                        last_indices.push(batch_idx);
                    }
                    batch_idx += 1;
                }
            }

            self.ctx
                .decode(&mut batch)
                .map_err(|e| BackendError::InferenceError(format!("decode: {}", e)))?;

            Ok(last_indices)
        }

        /// Create a sampler with temperature + top-p + distribution.
        pub fn make_sampler(temperature: f32, top_p: f32, seed: u32) -> LlamaSampler {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_p(top_p, 1),
                LlamaSampler::dist(seed),
            ])
        }

        /// Sample the next token using the context's logits.
        pub fn sample_next(
            &self,
            sampler: &mut LlamaSampler,
            batch_idx: i32,
        ) -> u32 {
            sampler.sample(&self.ctx, batch_idx).0 as u32
        }

        /// Get model config.
        pub fn model_config(&self) -> &ModelConfig {
            &self.config
        }
    }
}

#[cfg(feature = "llama-cpp")]
pub use inner::LlamaCppBackend;
