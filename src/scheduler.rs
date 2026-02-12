//! Sequence scheduler — orchestrates batched generation.
//!
//! The scheduler manages the lifecycle of sequences through the inference
//! pipeline: prefill → decode → finish. It decides which sequences to batch
//! together for each step and coordinates the KV cache.
//!
//! This is the "vLLM scheduler" equivalent — the piece that turns individual
//! attention kernel calls into a continuous generation loop.

use crate::pool::kv_cache::{KvCacheConfig, PagedKvCache, PageTable, SeqId};
use crate::pool::{PoolError, VramPool};
use std::collections::VecDeque;

/// The state of a sequence in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// Waiting to be scheduled (in the queue).
    Waiting,
    /// Currently generating tokens.
    Running,
    /// Finished (hit EOS or max length).
    Finished,
}

/// A generation request submitted to the scheduler.
#[derive(Debug)]
pub struct SeqRequest {
    /// Unique ID assigned by the scheduler.
    pub id: SeqId,
    /// Current state.
    pub state: SeqState,
    /// Input token IDs (prompt). Consumed during prefill.
    pub prompt_tokens: Vec<u32>,
    /// Generated token IDs so far.
    pub output_tokens: Vec<u32>,
    /// Maximum total tokens (prompt + output) before forced stop.
    pub max_tokens: u32,
    /// EOS token ID — generation stops when this is sampled.
    pub eos_token_id: u32,
    /// Number of prompt tokens already prefilled.
    pub prefill_pos: usize,
}

/// Output from a single scheduler step.
#[derive(Debug)]
pub struct SchedulerStep {
    /// Sequences in this batch (their SeqIds).
    pub batch_seq_ids: Vec<SeqId>,
    /// Page table for FlashInfer (CSR format).
    pub page_table: PageTable,
    /// Number of sequences in the batch.
    pub batch_size: usize,
}

/// The scheduler itself.
pub struct Scheduler {
    /// KV cache (shared across all sequences).
    kv_cache: PagedKvCache,
    /// All active requests, indexed by SeqId.
    requests: Vec<Option<SeqRequest>>,
    /// Queue of sequences waiting to start.
    waiting_queue: VecDeque<SeqId>,
    /// Currently running sequences.
    running: Vec<SeqId>,
    /// Maximum batch size (how many sequences to decode simultaneously).
    max_batch_size: usize,
    /// Next sequence ID to assign.
    next_id: u32,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(kv_config: KvCacheConfig, max_batch_size: usize) -> Self {
        Self {
            kv_cache: PagedKvCache::new(kv_config),
            requests: Vec::new(),
            waiting_queue: VecDeque::new(),
            running: Vec::new(),
            max_batch_size,
            next_id: 0,
        }
    }

    /// Submit a new generation request.
    ///
    /// Returns the SeqId which can be used to track the request.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: u32,
        eos_token_id: u32,
    ) -> SeqId {
        let id = SeqId(self.next_id);
        self.next_id += 1;

        let req = SeqRequest {
            id,
            state: SeqState::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_tokens,
            eos_token_id,
            prefill_pos: 0,
        };

        let idx = id.0 as usize;
        if idx >= self.requests.len() {
            self.requests.resize_with(idx + 1, || None);
        }
        self.requests[idx] = Some(req);
        self.waiting_queue.push_back(id);

        id
    }

    /// Schedule the next batch of sequences for decode.
    ///
    /// This is called once per generation step. It:
    /// 1. Promotes waiting sequences to running (if there's capacity)
    /// 2. Builds a batch from running sequences
    /// 3. Returns the page table for FlashInfer
    pub fn schedule(&mut self, pool: &mut VramPool) -> Result<Option<SchedulerStep>, PoolError> {
        // Promote waiting → running (up to max_batch_size)
        while self.running.len() < self.max_batch_size {
            if let Some(id) = self.waiting_queue.pop_front() {
                // Allocate KV cache pages for this sequence
                let kv_id = self.kv_cache.add_sequence(pool)?;
                // kv_id should match id since we're using the same counter pattern
                // but let's be defensive
                if let Some(req) = self.requests.get_mut(id.0 as usize).and_then(|r| r.as_mut()) {
                    req.state = SeqState::Running;
                }
                self.running.push(id);
            } else {
                break;
            }
        }

        if self.running.is_empty() {
            return Ok(None);
        }

        // Build page table from running sequences
        let page_table = self
            .kv_cache
            .build_page_table(&self.running)
            .ok_or(PoolError::InvalidPage(crate::pool::PageId(0)))?;

        Ok(Some(SchedulerStep {
            batch_seq_ids: self.running.clone(),
            page_table,
            batch_size: self.running.len(),
        }))
    }

    /// Record that new tokens were generated for the current batch.
    ///
    /// Called after each decode step with the sampled token for each sequence.
    /// Handles EOS detection and sequence completion.
    pub fn step(
        &mut self,
        sampled_tokens: &[(SeqId, u32)],
        pool: &mut VramPool,
    ) -> Result<Vec<SeqId>, PoolError> {
        let mut finished = Vec::new();

        for &(id, token) in sampled_tokens {
            let req = match self.requests.get_mut(id.0 as usize).and_then(|r| r.as_mut()) {
                Some(r) => r,
                None => continue,
            };

            req.output_tokens.push(token);

            // Update KV cache — one new token for this sequence
            self.kv_cache.append_tokens(id, 1, pool)?;

            // Check stopping conditions
            let total_tokens = req.prompt_tokens.len() + req.output_tokens.len();
            if token == req.eos_token_id || total_tokens as u32 >= req.max_tokens {
                req.state = SeqState::Finished;
                finished.push(id);

                // Free KV cache pages
                self.kv_cache.remove_sequence(id, pool)?;
            }
        }

        // Remove finished sequences from running list
        self.running.retain(|id| !finished.contains(id));

        Ok(finished)
    }

    /// Get the generated tokens for a sequence.
    pub fn get_output(&self, id: SeqId) -> Option<&[u32]> {
        self.requests
            .get(id.0 as usize)
            .and_then(|r| r.as_ref())
            .map(|r| r.output_tokens.as_slice())
    }

    /// Get the state of a sequence.
    pub fn get_state(&self, id: SeqId) -> Option<SeqState> {
        self.requests
            .get(id.0 as usize)
            .and_then(|r| r.as_ref())
            .map(|r| r.state)
    }

    /// Check if there are any active sequences (waiting or running).
    pub fn has_active(&self) -> bool {
        !self.running.is_empty() || !self.waiting_queue.is_empty()
    }

    /// Number of currently running sequences.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Number of sequences waiting in queue.
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Access the KV cache (for passing pool pointers to kernels).
    pub fn kv_cache(&self) -> &PagedKvCache {
        &self.kv_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::VramPoolConfig;
    use cudarc::driver::CudaContext;

    fn make_scheduler_and_pool() -> (Scheduler, VramPool) {
        let ctx = CudaContext::new(0).expect("CUDA required");
        let stream = ctx.default_stream();

        let kv_config = KvCacheConfig {
            num_kv_heads: 8,
            head_dim: 128,
            tokens_per_page: 16,
            dtype_bytes: 2,
        };

        let page_bytes = kv_config.k_page_bytes();
        let pool_config = VramPoolConfig {
            page_size_bytes: page_bytes,
            num_pages: 256,
        };

        let pool = VramPool::new(&stream, &pool_config).expect("Failed to create pool");
        let scheduler = Scheduler::new(kv_config, 4); // max 4 concurrent sequences

        (scheduler, pool)
    }

    #[test]
    fn add_and_schedule_single() {
        let (mut sched, mut pool) = make_scheduler_and_pool();

        let id = sched.add_request(vec![1, 2, 3], 100, 0);
        assert_eq!(sched.num_waiting(), 1);
        assert_eq!(sched.num_running(), 0);

        let step = sched.schedule(&mut pool).unwrap().unwrap();
        assert_eq!(step.batch_size, 1);
        assert_eq!(step.batch_seq_ids, vec![id]);
        assert_eq!(sched.num_running(), 1);
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn batch_multiple_sequences() {
        let (mut sched, mut pool) = make_scheduler_and_pool();

        let id1 = sched.add_request(vec![1, 2, 3], 100, 0);
        let id2 = sched.add_request(vec![4, 5, 6], 100, 0);
        let id3 = sched.add_request(vec![7, 8], 100, 0);

        let step = sched.schedule(&mut pool).unwrap().unwrap();
        assert_eq!(step.batch_size, 3);
        assert_eq!(step.batch_seq_ids, vec![id1, id2, id3]);
    }

    #[test]
    fn max_batch_size_respected() {
        let (mut sched, mut pool) = make_scheduler_and_pool();

        // Add 6 requests, but max batch is 4
        for i in 0..6 {
            sched.add_request(vec![i], 100, 0);
        }

        let step = sched.schedule(&mut pool).unwrap().unwrap();
        assert_eq!(step.batch_size, 4);
        assert_eq!(sched.num_waiting(), 2);
    }

    #[test]
    fn eos_finishes_sequence() {
        let (mut sched, mut pool) = make_scheduler_and_pool();

        let id = sched.add_request(vec![1, 2, 3], 100, 999); // EOS = 999
        sched.schedule(&mut pool).unwrap();

        // Generate a few tokens
        sched.step(&[(id, 10)], &mut pool).unwrap();
        sched.step(&[(id, 20)], &mut pool).unwrap();
        assert_eq!(sched.get_state(id), Some(SeqState::Running));

        // Hit EOS
        let finished = sched.step(&[(id, 999)], &mut pool).unwrap();
        assert_eq!(finished, vec![id]);
        assert_eq!(sched.get_state(id), Some(SeqState::Finished));
        assert_eq!(sched.get_output(id), Some([10, 20, 999].as_slice()));
        assert_eq!(sched.num_running(), 0);
    }

    #[test]
    fn max_tokens_finishes_sequence() {
        let (mut sched, mut pool) = make_scheduler_and_pool();

        // Prompt is 3 tokens, max total is 5, so only 2 output tokens before stop
        let id = sched.add_request(vec![1, 2, 3], 5, 999);
        sched.schedule(&mut pool).unwrap();

        sched.step(&[(id, 10)], &mut pool).unwrap();
        assert_eq!(sched.get_state(id), Some(SeqState::Running));

        let finished = sched.step(&[(id, 20)], &mut pool).unwrap();
        assert_eq!(finished, vec![id]);
        assert_eq!(sched.get_state(id), Some(SeqState::Finished));
    }

    #[test]
    fn finished_frees_pages_for_new_sequences() {
        let (mut sched, mut pool) = make_scheduler_and_pool();
        let initial_free = pool.free_count();

        let id1 = sched.add_request(vec![1], 10, 999);
        sched.schedule(&mut pool).unwrap();

        // Some pages allocated
        assert!(pool.free_count() < initial_free);

        // Finish sequence
        sched.step(&[(id1, 999)], &mut pool).unwrap();

        // Pages returned
        assert_eq!(pool.free_count(), initial_free);

        // New sequence can use them
        let id2 = sched.add_request(vec![2], 10, 999);
        sched.schedule(&mut pool).unwrap();
        assert_eq!(sched.num_running(), 1);
        assert_eq!(sched.get_state(id2), Some(SeqState::Running));
    }

    #[test]
    fn self_feeding_scenario() {
        // Simulates the multimodal self-feeding loop:
        // Seq 1: voice transcription (runs for a while, produces text)
        // Seq 2: polish pass (submitted mid-flight, batched with seq 1)
        let (mut sched, mut pool) = make_scheduler_and_pool();

        // Voice transcription starts
        let voice = sched.add_request(vec![100, 101, 102], 50, 999);
        sched.schedule(&mut pool).unwrap();
        assert_eq!(sched.num_running(), 1);

        // Generate some transcription tokens
        for i in 0..5 {
            sched.step(&[(voice, 200 + i)], &mut pool).unwrap();
        }

        // Mid-generation, submit polish request (batched with voice)
        let polish = sched.add_request(vec![300, 301], 30, 999);
        let step = sched.schedule(&mut pool).unwrap().unwrap();
        assert_eq!(step.batch_size, 2); // Both sequences batched!
        assert_eq!(sched.num_running(), 2);

        // Both generate in parallel
        sched
            .step(&[(voice, 210), (polish, 400)], &mut pool)
            .unwrap();
        sched
            .step(&[(voice, 211), (polish, 401)], &mut pool)
            .unwrap();

        // Voice finishes (EOS)
        let finished = sched
            .step(&[(voice, 999), (polish, 402)], &mut pool)
            .unwrap();
        assert_eq!(finished, vec![voice]);
        assert_eq!(sched.num_running(), 1); // Polish still going

        // Polish finishes
        let finished = sched.step(&[(polish, 999)], &mut pool).unwrap();
        assert_eq!(finished, vec![polish]);
        assert_eq!(sched.num_running(), 0);

        // Check outputs
        assert_eq!(
            sched.get_output(voice),
            Some([200, 201, 202, 203, 204, 210, 211, 999].as_slice())
        );
        assert_eq!(
            sched.get_output(polish),
            Some([400, 401, 402, 999].as_slice())
        );
    }
}
