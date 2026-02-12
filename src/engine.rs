//! Generation engine — the high-level API for multi-sequence text generation.
//!
//! Wires the llama.cpp backend to sequence lifecycle management. You submit
//! prompts, call step() to advance generation, and collect outputs.
//!
//! ```ignore
//! let mut engine = GenerationEngine::new(backend, Default::default());
//! let s1 = engine.submit("The capital of France is", 64)?;
//! let s2 = engine.submit("fn main() {", 64)?;
//!
//! while engine.has_active() {
//!     let new_tokens = engine.step()?;
//!     // new_tokens: Vec<(SeqId, u32)> — stream them, log them, whatever
//! }
//!
//! let text = engine.get_text(s1)?;
//! ```

#[cfg(feature = "llama-cpp")]
mod inner {
    use crate::backend::llama_cpp::LlamaCppBackend;
    use crate::backend::BackendError;
    use llama_cpp_2::sampling::LlamaSampler;

    /// Unique sequence identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SeqId(pub u32);

    /// Sampling configuration for generation.
    #[derive(Debug, Clone)]
    pub struct SamplingConfig {
        pub temperature: f32,
        pub top_p: f32,
        pub seed: u32,
    }

    impl Default for SamplingConfig {
        fn default() -> Self {
            Self {
                temperature: 0.7,
                top_p: 0.9,
                seed: 42,
            }
        }
    }

    /// Engine configuration.
    #[derive(Debug, Clone)]
    pub struct EngineConfig {
        pub sampling: SamplingConfig,
        pub max_batch_size: usize,
    }

    impl Default for EngineConfig {
        fn default() -> Self {
            Self {
                sampling: SamplingConfig::default(),
                max_batch_size: 8,
            }
        }
    }

    /// Phase of a sequence's lifecycle.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Phase {
        /// Prompt tokens haven't been fed yet.
        NeedsPrefill,
        /// Actively generating tokens.
        Decoding,
        /// Hit EOS or max_tokens.
        Finished,
    }

    /// Internal state for a single sequence.
    struct SeqState {
        prompt_tokens: Vec<u32>,
        output_tokens: Vec<u32>,
        max_tokens: u32,
        eos_token_id: u32,
        position: i32,
        phase: Phase,
        /// The most recently sampled token, waiting to be forwarded.
        next_token: Option<u32>,
    }

    /// The generation engine.
    ///
    /// Manages multiple concurrent sequences through a shared llama.cpp backend.
    /// Handles prefill, decode, sampling, and sequence lifecycle automatically.
    pub struct GenerationEngine {
        backend: LlamaCppBackend,
        sequences: Vec<Option<SeqState>>,
        samplers: Vec<Option<LlamaSampler>>,
        config: EngineConfig,
        next_id: u32,
        // Reusable buffers — cleared and reused each step() to avoid allocation
        buf_generated: Vec<(SeqId, u32)>,
        buf_batch_entries: Vec<(u32, i32, i32)>,
        buf_batch_indices: Vec<usize>,
    }

    impl GenerationEngine {
        /// Create a new engine wrapping a loaded backend.
        pub fn new(backend: LlamaCppBackend, config: EngineConfig) -> Self {
            let cap = config.max_batch_size;
            Self {
                backend,
                sequences: Vec::new(),
                samplers: Vec::new(),
                config,
                next_id: 0,
                buf_generated: Vec::with_capacity(cap),
                buf_batch_entries: Vec::with_capacity(cap),
                buf_batch_indices: Vec::with_capacity(cap),
            }
        }

        /// Submit a prompt for generation.
        ///
        /// Returns a SeqId for tracking. The sequence will be prefilled and
        /// start generating on the next `step()` call.
        pub fn submit(
            &mut self,
            prompt: &str,
            max_tokens: u32,
        ) -> Result<SeqId, BackendError> {
            let tokens = self.backend.tokenize_str(prompt, true)?;
            self.submit_tokens(tokens, max_tokens)
        }

        /// Submit pre-tokenized prompt for generation.
        pub fn submit_tokens(
            &mut self,
            tokens: Vec<u32>,
            max_tokens: u32,
        ) -> Result<SeqId, BackendError> {
            let id = SeqId(self.next_id);
            self.next_id += 1;
            let idx = id.0 as usize;

            let state = SeqState {
                prompt_tokens: tokens,
                output_tokens: Vec::new(),
                max_tokens,
                eos_token_id: self.backend.eos_token(),
                position: 0,
                phase: Phase::NeedsPrefill,
                next_token: None,
            };

            // Use unique seed per sequence for sampling diversity
            let sampler = LlamaCppBackend::make_sampler(
                self.config.sampling.temperature,
                self.config.sampling.top_p,
                self.config.sampling.seed.wrapping_add(id.0),
            );

            if idx >= self.sequences.len() {
                self.sequences.resize_with(idx + 1, || None);
                self.samplers.resize_with(idx + 1, || None);
            }
            self.sequences[idx] = Some(state);
            self.samplers[idx] = Some(sampler);

            Ok(id)
        }

        /// Advance all active sequences by one step.
        ///
        /// Returns newly generated tokens: `(SeqId, token_id)` pairs.
        /// Handles prefill automatically — sequences that haven't been prefilled
        /// yet will be prefilled before their first decode step.
        pub fn step(&mut self) -> Result<Vec<(SeqId, u32)>, BackendError> {
            self.buf_generated.clear();
            self.buf_batch_entries.clear();
            self.buf_batch_indices.clear();

            // Single scan: partition sequences into prefill vs decode
            // We collect indices into buf_batch_indices temporarily for prefill,
            // then reuse it for decode batch tracking.
            let mut prefill_count = 0usize;
            for (i, slot) in self.sequences.iter().enumerate() {
                if let Some(s) = slot {
                    match s.phase {
                        Phase::NeedsPrefill => {
                            // Pack prefill indices at the front of buf_batch_indices
                            self.buf_batch_indices.push(i);
                            prefill_count += 1;
                        }
                        _ => {}
                    }
                }
            }

            // Prefill — batched into one decode call
            if prefill_count > 0 {
                // Build refs directly from sequence data (no clone needed).
                // We temporarily split borrows: read sequences to build refs,
                // then call backend, then write back.
                let refs: Vec<(&[u32], i32)> = self.buf_batch_indices[..prefill_count]
                    .iter()
                    .map(|&idx| {
                        let seq = self.sequences[idx].as_ref().unwrap();
                        (seq.prompt_tokens.as_slice(), idx as i32)
                    })
                    .collect();

                let last_indices = self.backend.forward_prefill_batch(&refs)?;

                for (i, &idx) in self.buf_batch_indices[..prefill_count].iter().enumerate() {
                    let sampler = self.samplers[idx].as_mut().unwrap();
                    let first_token = self.backend.sample_next(sampler, last_indices[i]);

                    let seq = self.sequences[idx].as_mut().unwrap();
                    seq.position = seq.prompt_tokens.len() as i32;
                    seq.phase = Phase::Decoding;
                    seq.next_token = Some(first_token);
                }
            }

            // Reset buf_batch_indices for decode phase
            self.buf_batch_indices.clear();

            // Decode: record tokens, check stopping, build forward batch
            for i in 0..self.sequences.len() {
                let should_decode = self.sequences[i]
                    .as_ref()
                    .map(|s| s.phase == Phase::Decoding && s.next_token.is_some())
                    .unwrap_or(false);

                if !should_decode {
                    continue;
                }

                let seq = self.sequences[i].as_mut().unwrap();
                let token = seq.next_token.take().unwrap();

                seq.output_tokens.push(token);
                self.buf_generated.push((SeqId(i as u32), token));

                let total = seq.prompt_tokens.len() + seq.output_tokens.len();
                if token == seq.eos_token_id || total as u32 >= seq.max_tokens {
                    seq.phase = Phase::Finished;
                    continue;
                }

                let pos = seq.position;
                self.buf_batch_entries.push((token, i as i32, pos));
                self.buf_batch_indices.push(i);
                seq.position += 1;
            }

            // Single batched forward pass for all surviving sequences
            if !self.buf_batch_entries.is_empty() {
                self.backend.forward_batch(&self.buf_batch_entries)?;

                for (i, &idx) in self.buf_batch_indices.iter().enumerate() {
                    let sampler = self.samplers[idx].as_mut().unwrap();
                    let next = self.backend.sample_next(sampler, i as i32);
                    self.sequences[idx].as_mut().unwrap().next_token = Some(next);
                }
            }

            let mut result = Vec::new();
            std::mem::swap(&mut result, &mut self.buf_generated);
            Ok(result)
        }

        /// Check if any sequences are still active.
        pub fn has_active(&self) -> bool {
            self.sequences.iter().any(|s| {
                s.as_ref()
                    .map(|s| s.phase != Phase::Finished)
                    .unwrap_or(false)
            })
        }

        /// Get the generated token IDs for a sequence.
        pub fn get_output(&self, id: SeqId) -> Option<&[u32]> {
            self.sequences
                .get(id.0 as usize)
                .and_then(|s| s.as_ref())
                .map(|s| s.output_tokens.as_slice())
        }

        /// Get the generated text for a sequence.
        pub fn get_text(&self, id: SeqId) -> Result<Option<String>, BackendError> {
            match self.get_output(id) {
                Some(tokens) => Ok(Some(self.backend.detokenize_tokens(tokens)?)),
                None => Ok(None),
            }
        }

        /// Check if a sequence has finished.
        pub fn is_finished(&self, id: SeqId) -> bool {
            self.sequences
                .get(id.0 as usize)
                .and_then(|s| s.as_ref())
                .map(|s| s.phase == Phase::Finished)
                .unwrap_or(true)
        }

        /// Get number of active (non-finished) sequences.
        pub fn num_active(&self) -> usize {
            self.sequences
                .iter()
                .filter(|s| {
                    s.as_ref()
                        .map(|s| s.phase != Phase::Finished)
                        .unwrap_or(false)
                })
                .count()
        }

        /// Detokenize a slice of token IDs.
        pub fn detokenize(&self, tokens: &[u32]) -> Result<String, BackendError> {
            self.backend.detokenize_tokens(tokens)
        }

        /// Get the underlying model config.
        pub fn model_config(&self) -> &crate::backend::ModelConfig {
            self.backend.model_config()
        }
    }
}

#[cfg(feature = "llama-cpp")]
pub use inner::*;
