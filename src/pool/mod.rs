//! VRAM page pool — the foundation of gpu-infer.
//!
//! All GPU memory flows through here. The pool pre-allocates a VRAM budget
//! as fixed-size pages and hands them out on request. When pages are freed,
//! they return to the pool for reuse. No fragmentation, no per-model reservations.
//!
//! Page size is configurable but typically matches the KV cache block size
//! used by the attention kernels (e.g., 16 or 32 tokens worth of KV data).

pub mod kv_cache;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;

/// Unique identifier for an allocated page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

/// Errors from the VRAM pool.
#[derive(Debug, Error)]
pub enum PoolError {
    #[error("out of VRAM pages ({requested} requested, {available} available, {total} total)")]
    OutOfPages {
        requested: usize,
        available: usize,
        total: usize,
    },

    #[error("invalid page ID: {0:?}")]
    InvalidPage(PageId),

    #[error("page {0:?} is not currently allocated")]
    NotAllocated(PageId),

    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),
}

/// Configuration for the VRAM pool.
#[derive(Debug, Clone)]
pub struct VramPoolConfig {
    /// Size of each page in bytes.
    pub page_size_bytes: usize,

    /// Total number of pages to allocate.
    pub num_pages: usize,
}

impl VramPoolConfig {
    /// Create a config from a total VRAM budget and page size.
    ///
    /// `budget_bytes`: Total VRAM to reserve for the pool.
    /// `page_size_bytes`: Size of each page. Typically derived from
    ///   `page_size_tokens * num_kv_heads * head_dim * 2 (K+V) * sizeof(fp16)`.
    pub fn from_budget(budget_bytes: usize, page_size_bytes: usize) -> Self {
        Self {
            page_size_bytes,
            num_pages: budget_bytes / page_size_bytes,
        }
    }

    /// Helper: compute page size for a KV cache with given dimensions.
    ///
    /// Returns bytes per page for both K and V combined.
    pub fn kv_page_bytes(
        tokens_per_page: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> usize {
        // Each page holds K and V: 2 * tokens * heads * dim * dtype
        2 * tokens_per_page * num_kv_heads * head_dim * dtype_bytes
    }
}

/// A pool of fixed-size VRAM pages.
///
/// The pool owns one large contiguous GPU allocation, subdivided into pages.
/// Pages are allocated and freed individually without CUDA malloc/free overhead.
pub struct VramPool {
    /// The single GPU allocation backing all pages.
    _backing: CudaSlice<u8>,

    /// Base device pointer of the backing allocation.
    base_ptr: u64,

    /// Size of each page in bytes.
    page_size_bytes: usize,

    /// Total number of pages.
    num_pages: usize,

    /// Free page IDs, ready to allocate.
    free_list: VecDeque<PageId>,

    /// Tracks which pages are currently allocated (true = allocated).
    allocated: Vec<bool>,

    /// CUDA stream used for the backing allocation.
    stream: Arc<CudaStream>,
}

impl VramPool {
    /// Create a new VRAM pool.
    ///
    /// Allocates `config.num_pages * config.page_size_bytes` bytes of GPU memory
    /// as a single contiguous allocation.
    pub fn new(stream: &Arc<CudaStream>, config: &VramPoolConfig) -> Result<Self, PoolError> {
        let total_bytes = config.num_pages * config.page_size_bytes;

        // Single large allocation — no fragmentation possible
        let backing: CudaSlice<u8> = stream.alloc_zeros(total_bytes)?;

        // Get the base pointer
        let (base_ptr, _guard) = backing.device_ptr(stream);
        let base_ptr = base_ptr as u64;
        drop(_guard);

        // Initialize free list with all pages
        let free_list: VecDeque<PageId> = (0..config.num_pages as u32).map(PageId).collect();
        let allocated = vec![false; config.num_pages];

        Ok(Self {
            _backing: backing,
            base_ptr,
            page_size_bytes: config.page_size_bytes,
            num_pages: config.num_pages,
            free_list,
            allocated,
            stream: stream.clone(),
        })
    }

    /// Allocate a single page. Returns its ID.
    pub fn alloc_page(&mut self) -> Result<PageId, PoolError> {
        let id = self.free_list.pop_front().ok_or(PoolError::OutOfPages {
            requested: 1,
            available: 0,
            total: self.num_pages,
        })?;
        self.allocated[id.0 as usize] = true;
        Ok(id)
    }

    /// Allocate `n` contiguous pages. Returns their IDs.
    ///
    /// Note: pages are logically contiguous (sequential IDs) but callers
    /// should not assume physical contiguity — use `page_ptr()` per page.
    pub fn alloc_pages(&mut self, n: usize) -> Result<Vec<PageId>, PoolError> {
        if self.free_list.len() < n {
            return Err(PoolError::OutOfPages {
                requested: n,
                available: self.free_list.len(),
                total: self.num_pages,
            });
        }
        let mut pages = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_list.pop_front().unwrap();
            self.allocated[id.0 as usize] = true;
            pages.push(id);
        }
        Ok(pages)
    }

    /// Free a page, returning it to the pool.
    pub fn free_page(&mut self, id: PageId) -> Result<(), PoolError> {
        let idx = id.0 as usize;
        if idx >= self.num_pages {
            return Err(PoolError::InvalidPage(id));
        }
        if !self.allocated[idx] {
            return Err(PoolError::NotAllocated(id));
        }
        self.allocated[idx] = false;
        self.free_list.push_back(id);
        Ok(())
    }

    /// Free multiple pages at once.
    pub fn free_pages(&mut self, ids: &[PageId]) -> Result<(), PoolError> {
        for &id in ids {
            self.free_page(id)?;
        }
        Ok(())
    }

    /// Get the raw device pointer for a page.
    ///
    /// This is what you pass to CUDA kernels. The pointer is valid for
    /// `page_size_bytes` bytes starting at the returned address.
    pub fn page_ptr(&self, id: PageId) -> Result<u64, PoolError> {
        let idx = id.0 as usize;
        if idx >= self.num_pages {
            return Err(PoolError::InvalidPage(id));
        }
        if !self.allocated[idx] {
            return Err(PoolError::NotAllocated(id));
        }
        Ok(self.base_ptr + (idx * self.page_size_bytes) as u64)
    }

    /// Number of free pages available.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of allocated pages.
    pub fn allocated_count(&self) -> usize {
        self.num_pages - self.free_list.len()
    }

    /// Total number of pages in the pool.
    pub fn total_count(&self) -> usize {
        self.num_pages
    }

    /// Page size in bytes.
    pub fn page_size_bytes(&self) -> usize {
        self.page_size_bytes
    }

    /// Total pool size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.num_pages * self.page_size_bytes
    }

    /// Allocated bytes.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_count() * self.page_size_bytes
    }

    /// Free bytes.
    pub fn free_bytes(&self) -> usize {
        self.free_count() * self.page_size_bytes
    }

    /// Utilization as a fraction (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        self.allocated_count() as f64 / self.num_pages as f64
    }

    /// Base device pointer of the entire pool allocation.
    ///
    /// Used by the KV cache to compute page addresses for FlashInfer.
    /// The pool is one contiguous allocation; page N starts at
    /// `base_ptr() + N * page_size_bytes()`.
    pub fn base_ptr(&self) -> u64 {
        self.base_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(num_pages: usize, page_size: usize) -> VramPool {
        let ctx = CudaContext::new(0).expect("CUDA required for tests");
        let stream = ctx.default_stream();
        let config = VramPoolConfig {
            page_size_bytes: page_size,
            num_pages,
        };
        VramPool::new(&stream, &config).expect("Failed to create pool")
    }

    #[test]
    fn alloc_and_free_single_page() {
        let mut pool = make_pool(16, 4096);
        assert_eq!(pool.free_count(), 16);

        let page = pool.alloc_page().unwrap();
        assert_eq!(pool.free_count(), 15);
        assert_eq!(pool.allocated_count(), 1);

        let ptr = pool.page_ptr(page).unwrap();
        assert!(ptr > 0);

        pool.free_page(page).unwrap();
        assert_eq!(pool.free_count(), 16);
    }

    #[test]
    fn alloc_multiple_pages() {
        let mut pool = make_pool(64, 4096);
        let pages = pool.alloc_pages(10).unwrap();
        assert_eq!(pages.len(), 10);
        assert_eq!(pool.free_count(), 54);

        // Each page has a distinct pointer
        let ptrs: Vec<u64> = pages.iter().map(|&p| pool.page_ptr(p).unwrap()).collect();
        for i in 0..ptrs.len() {
            for j in (i + 1)..ptrs.len() {
                assert_ne!(ptrs[i], ptrs[j], "pages {} and {} have same ptr", i, j);
            }
        }

        pool.free_pages(&pages).unwrap();
        assert_eq!(pool.free_count(), 64);
    }

    #[test]
    fn exhaust_pool() {
        let mut pool = make_pool(4, 4096);
        let _pages = pool.alloc_pages(4).unwrap();
        assert_eq!(pool.free_count(), 0);

        let err = pool.alloc_page().unwrap_err();
        assert!(matches!(err, PoolError::OutOfPages { .. }));
    }

    #[test]
    fn double_free_fails() {
        let mut pool = make_pool(4, 4096);
        let page = pool.alloc_page().unwrap();
        pool.free_page(page).unwrap();

        let err = pool.free_page(page).unwrap_err();
        assert!(matches!(err, PoolError::NotAllocated(_)));
    }

    #[test]
    fn page_ptr_offsets_are_correct() {
        let page_size = 4096;
        let mut pool = make_pool(8, page_size);

        let p0 = pool.alloc_page().unwrap();
        let p1 = pool.alloc_page().unwrap();

        let ptr0 = pool.page_ptr(p0).unwrap();
        let ptr1 = pool.page_ptr(p1).unwrap();

        // Pages should be exactly page_size apart (since IDs are sequential)
        assert_eq!(ptr1 - ptr0, page_size as u64);
    }

    #[test]
    fn from_budget_calculates_correctly() {
        let page_bytes = VramPoolConfig::kv_page_bytes(
            16,  // tokens per page
            8,   // num_kv_heads
            128, // head_dim
            2,   // fp16
        );
        // 2 * 16 * 8 * 128 * 2 = 65536 bytes = 64 KB per page
        assert_eq!(page_bytes, 65536);

        let config = VramPoolConfig::from_budget(1024 * 1024, page_bytes); // 1 MB budget
        assert_eq!(config.num_pages, 16); // 1MB / 64KB = 16 pages
    }

    #[test]
    fn utilization_tracking() {
        let mut pool = make_pool(100, 1024);
        assert_eq!(pool.utilization(), 0.0);

        let _pages = pool.alloc_pages(50).unwrap();
        assert!((pool.utilization() - 0.5).abs() < f64::EPSILON);

        assert_eq!(pool.allocated_bytes(), 50 * 1024);
        assert_eq!(pool.free_bytes(), 50 * 1024);
        assert_eq!(pool.total_bytes(), 100 * 1024);
    }
}
