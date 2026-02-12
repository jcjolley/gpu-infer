//! Paged KV cache built on top of the VRAM pool.
//!
//! Each sequence gets its own page table — a list of page IDs from the pool.
//! As tokens are generated, new pages are allocated on demand. When a sequence
//! finishes, its pages return to the pool for reuse by other sequences.

use super::{PageId, PoolError, VramPool};

/// Identifies a sequence within the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub u32);

/// Configuration for paged KV cache dimensions.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of KV heads (for GQA, this is the smaller number).
    pub num_kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Number of tokens stored per page.
    pub tokens_per_page: u32,
    /// Bytes per element (2 for fp16, 1 for fp8).
    pub dtype_bytes: u32,
}

impl KvCacheConfig {
    /// Bytes needed per page for K cache only.
    pub fn k_page_bytes(&self) -> usize {
        (self.tokens_per_page * self.num_kv_heads * self.head_dim * self.dtype_bytes) as usize
    }

    /// Bytes needed per page for V cache only.
    pub fn v_page_bytes(&self) -> usize {
        self.k_page_bytes() // Same size as K
    }

    /// Total bytes per page (K + V).
    pub fn total_page_bytes(&self) -> usize {
        self.k_page_bytes() + self.v_page_bytes()
    }
}

/// A single sequence's page table and metadata.
#[derive(Debug)]
struct SequenceState {
    /// Pages allocated for K cache, in order.
    k_pages: Vec<PageId>,
    /// Pages allocated for V cache, in order.
    v_pages: Vec<PageId>,
    /// Number of tokens currently stored.
    num_tokens: u32,
}

/// Paged KV cache managing multiple sequences over a shared VRAM pool.
///
/// The KV cache uses two separate page pools — one for K, one for V —
/// matching FlashInfer's expected layout where k_data and v_data are
/// separate contiguous allocations indexed by page table.
pub struct PagedKvCache {
    config: KvCacheConfig,
    sequences: Vec<Option<SequenceState>>,
    next_seq_id: u32,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    pub fn new(config: KvCacheConfig) -> Self {
        Self {
            config,
            sequences: Vec::new(),
            next_seq_id: 0,
        }
    }

    /// Register a new sequence, allocating initial pages for it.
    pub fn add_sequence(&mut self, pool: &mut VramPool) -> Result<SeqId, PoolError> {
        let id = SeqId(self.next_seq_id);
        self.next_seq_id += 1;

        // Allocate one initial page for K and one for V
        let k_page = pool.alloc_page()?;
        let v_page = pool.alloc_page()?;

        let state = SequenceState {
            k_pages: vec![k_page],
            v_pages: vec![v_page],
            num_tokens: 0,
        };

        let idx = id.0 as usize;
        if idx >= self.sequences.len() {
            self.sequences.resize_with(idx + 1, || None);
        }
        self.sequences[idx] = Some(state);

        Ok(id)
    }

    /// Remove a sequence, freeing all its pages back to the pool.
    pub fn remove_sequence(&mut self, id: SeqId, pool: &mut VramPool) -> Result<(), PoolError> {
        let idx = id.0 as usize;
        let state = self.sequences.get_mut(idx)
            .and_then(|s| s.take())
            .ok_or(PoolError::InvalidPage(PageId(id.0)))?;

        for page in &state.k_pages {
            pool.free_page(*page)?;
        }
        for page in &state.v_pages {
            pool.free_page(*page)?;
        }

        Ok(())
    }

    /// Record that `n` new tokens were added to a sequence.
    /// Allocates new pages if the current pages are full.
    pub fn append_tokens(
        &mut self,
        id: SeqId,
        n: u32,
        pool: &mut VramPool,
    ) -> Result<(), PoolError> {
        let idx = id.0 as usize;
        let tokens = self.sequences.get(idx)
            .and_then(|s| s.as_ref())
            .ok_or(PoolError::InvalidPage(PageId(id.0)))?
            .num_tokens;

        let new_total = tokens + n;
        let pages_needed = self.pages_for_tokens(new_total);

        let state = self.sequences.get_mut(idx)
            .and_then(|s| s.as_mut())
            .unwrap(); // safe: we just checked above
        let current_pages = state.k_pages.len();

        // Allocate additional pages if needed
        for _ in current_pages..pages_needed {
            let k_page = pool.alloc_page()?;
            let v_page = pool.alloc_page()?;
            state.k_pages.push(k_page);
            state.v_pages.push(v_page);
        }

        state.num_tokens = new_total;
        Ok(())
    }

    /// Get the number of tokens in a sequence.
    pub fn seq_len(&self, id: SeqId) -> Option<u32> {
        self.sequences.get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.num_tokens)
    }

    /// Get the page IDs for a sequence's K cache.
    pub fn k_page_ids(&self, id: SeqId) -> Option<&[PageId]> {
        self.sequences.get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.k_pages.as_slice())
    }

    /// Get the page IDs for a sequence's V cache.
    pub fn v_page_ids(&self, id: SeqId) -> Option<&[PageId]> {
        self.sequences.get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.v_pages.as_slice())
    }

    /// Number of tokens that fit in the last page of a sequence.
    /// This is what FlashInfer's `last_page_len` parameter needs.
    pub fn last_page_len(&self, id: SeqId) -> Option<u32> {
        let tokens = self.seq_len(id)?;
        if tokens == 0 {
            Some(0)
        } else {
            let remainder = tokens % self.config.tokens_per_page;
            Some(if remainder == 0 { self.config.tokens_per_page } else { remainder })
        }
    }

    /// Build the CSR-format arrays that FlashInfer expects for a batch of sequences.
    ///
    /// Returns (kv_indices, kv_indptr, kv_last_page_len) suitable for passing
    /// to `flashinfer_batch_decode_paged_f16`.
    ///
    /// Note: K and V use separate page pools, so kv_indices contains K page indices.
    /// V page indices are separate (same structure, different pool).
    pub fn build_page_table(&self, seq_ids: &[SeqId]) -> Option<PageTable> {
        let mut kv_indices_k: Vec<i32> = Vec::new();
        let mut kv_indices_v: Vec<i32> = Vec::new();
        let mut kv_indptr: Vec<i32> = vec![0];
        let mut last_page_lens: Vec<i32> = Vec::new();

        for &id in seq_ids {
            let state = self.sequences.get(id.0 as usize)?.as_ref()?;

            for page in &state.k_pages {
                kv_indices_k.push(page.0 as i32);
            }
            for page in &state.v_pages {
                kv_indices_v.push(page.0 as i32);
            }

            kv_indptr.push(kv_indices_k.len() as i32);
            last_page_lens.push(self.last_page_len(id)? as i32);
        }

        Some(PageTable {
            k_indices: kv_indices_k,
            v_indices: kv_indices_v,
            indptr: kv_indptr,
            last_page_len: last_page_lens,
        })
    }

    /// Total pages currently allocated across all sequences.
    pub fn total_pages_allocated(&self) -> usize {
        self.sequences.iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.k_pages.len() + s.v_pages.len())
            .sum()
    }

    /// Number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.iter().filter(|s| s.is_some()).count()
    }

    /// Tokens per page.
    pub fn tokens_per_page(&self) -> u32 {
        self.config.tokens_per_page
    }

    fn pages_for_tokens(&self, tokens: u32) -> usize {
        if tokens == 0 {
            1 // Always have at least one page
        } else {
            ((tokens + self.config.tokens_per_page - 1) / self.config.tokens_per_page) as usize
        }
    }
}

/// CSR-format page table for FlashInfer batch decode.
#[derive(Debug)]
pub struct PageTable {
    /// K page indices for all sequences, concatenated.
    pub k_indices: Vec<i32>,
    /// V page indices for all sequences, concatenated.
    pub v_indices: Vec<i32>,
    /// CSR offsets: indptr[i] is the start of sequence i's pages in indices.
    pub indptr: Vec<i32>,
    /// Number of valid tokens in the last page of each sequence.
    pub last_page_len: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::VramPoolConfig;
    use cudarc::driver::CudaContext;

    fn make_pool_and_cache() -> (VramPool, PagedKvCache) {
        let ctx = CudaContext::new(0).expect("CUDA required");
        let stream = ctx.default_stream();

        let kv_config = KvCacheConfig {
            num_kv_heads: 8,
            head_dim: 128,
            tokens_per_page: 16,
            dtype_bytes: 2, // fp16
        };

        // Each K or V page: 16 * 8 * 128 * 2 = 32768 bytes = 32 KB
        let page_bytes = kv_config.k_page_bytes();
        assert_eq!(page_bytes, 32768);

        // 256 pages = 8 MB pool (enough for testing)
        let pool_config = VramPoolConfig {
            page_size_bytes: page_bytes,
            num_pages: 256,
        };
        let pool = VramPool::new(&stream, &pool_config).expect("Failed to create pool");
        let cache = PagedKvCache::new(kv_config);

        (pool, cache)
    }

    #[test]
    fn add_and_remove_sequence() {
        let (mut pool, mut cache) = make_pool_and_cache();
        let initial_free = pool.free_count();

        let seq = cache.add_sequence(&mut pool).unwrap();
        assert_eq!(cache.num_sequences(), 1);
        assert_eq!(cache.seq_len(seq), Some(0));
        // 2 pages allocated (1 K + 1 V)
        assert_eq!(pool.free_count(), initial_free - 2);

        cache.remove_sequence(seq, &mut pool).unwrap();
        assert_eq!(cache.num_sequences(), 0);
        assert_eq!(pool.free_count(), initial_free);
    }

    #[test]
    fn append_tokens_grows_pages() {
        let (mut pool, mut cache) = make_pool_and_cache();

        let seq = cache.add_sequence(&mut pool).unwrap();
        // Start with 1 K page + 1 V page = 2 pages
        assert_eq!(cache.total_pages_allocated(), 2);

        // Append 10 tokens — fits in first page (16 tokens/page)
        cache.append_tokens(seq, 10, &mut pool).unwrap();
        assert_eq!(cache.seq_len(seq), Some(10));
        assert_eq!(cache.total_pages_allocated(), 2);

        // Append 10 more = 20 total — needs 2 pages each
        cache.append_tokens(seq, 10, &mut pool).unwrap();
        assert_eq!(cache.seq_len(seq), Some(20));
        assert_eq!(cache.total_pages_allocated(), 4); // 2 K + 2 V
    }

    #[test]
    fn last_page_len_correct() {
        let (mut pool, mut cache) = make_pool_and_cache();
        let seq = cache.add_sequence(&mut pool).unwrap();

        assert_eq!(cache.last_page_len(seq), Some(0));

        cache.append_tokens(seq, 10, &mut pool).unwrap();
        assert_eq!(cache.last_page_len(seq), Some(10));

        cache.append_tokens(seq, 6, &mut pool).unwrap(); // 16 total = full page
        assert_eq!(cache.last_page_len(seq), Some(16));

        cache.append_tokens(seq, 1, &mut pool).unwrap(); // 17 total = 1 in second page
        assert_eq!(cache.last_page_len(seq), Some(1));
    }

    #[test]
    fn build_page_table_single_seq() {
        let (mut pool, mut cache) = make_pool_and_cache();
        let seq = cache.add_sequence(&mut pool).unwrap();
        cache.append_tokens(seq, 40, &mut pool).unwrap(); // 3 pages needed

        let table = cache.build_page_table(&[seq]).unwrap();
        assert_eq!(table.k_indices.len(), 3);
        assert_eq!(table.v_indices.len(), 3);
        assert_eq!(table.indptr, vec![0, 3]);
        assert_eq!(table.last_page_len, vec![8]); // 40 % 16 = 8
    }

    #[test]
    fn build_page_table_multi_seq() {
        let (mut pool, mut cache) = make_pool_and_cache();

        let seq1 = cache.add_sequence(&mut pool).unwrap();
        cache.append_tokens(seq1, 20, &mut pool).unwrap(); // 2 pages

        let seq2 = cache.add_sequence(&mut pool).unwrap();
        cache.append_tokens(seq2, 50, &mut pool).unwrap(); // 4 pages (ceil(50/16))

        let table = cache.build_page_table(&[seq1, seq2]).unwrap();
        assert_eq!(table.indptr, vec![0, 2, 6]); // seq1: 2 pages, seq2: 4 pages
        assert_eq!(table.last_page_len, vec![4, 2]); // 20%16=4, 50%16=2
        assert_eq!(table.k_indices.len(), 6);
    }

    #[test]
    fn multiple_sequences_share_pool() {
        let (mut pool, mut cache) = make_pool_and_cache();
        let initial_free = pool.free_count();

        let seq1 = cache.add_sequence(&mut pool).unwrap();
        let seq2 = cache.add_sequence(&mut pool).unwrap();
        let seq3 = cache.add_sequence(&mut pool).unwrap();

        cache.append_tokens(seq1, 100, &mut pool).unwrap(); // 7 pages
        cache.append_tokens(seq2, 16, &mut pool).unwrap();  // 1 page
        cache.append_tokens(seq3, 5, &mut pool).unwrap();   // 1 page

        let total = cache.total_pages_allocated();
        // seq1: 7K + 7V = 14, seq2: 1K + 1V = 2, seq3: 1K + 1V = 2
        assert_eq!(total, 18);
        assert_eq!(pool.free_count(), initial_free - 18);

        // Remove seq1, its pages return to pool
        cache.remove_sequence(seq1, &mut pool).unwrap();
        assert_eq!(pool.free_count(), initial_free - 4);

        // seq2 and seq3 still healthy
        assert_eq!(cache.seq_len(seq2), Some(16));
        assert_eq!(cache.seq_len(seq3), Some(5));
    }
}
