//! gpu-infer: Shared GPU inference runtime.
//!
//! Provides a unified VRAM pool, paged KV cache, and attention kernel
//! integration for running multiple models on a single GPU efficiently.

pub mod attention;
pub mod backend;
pub mod engine;
pub mod pool;
pub mod scheduler;

pub use pool::{PageId, VramPool, VramPoolConfig};
pub use scheduler::Scheduler;
