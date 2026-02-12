//! Attention kernel wrappers (FlashInfer FFI).
//!
//! Provides safe Rust APIs around FlashInfer's CUDA attention kernels.

/// Raw FFI bindings to the C wrapper around FlashInfer.
pub mod ffi {
    extern "C" {
        /// Single-sequence decode attention with contiguous KV cache (fp16).
        pub fn flashinfer_single_decode_f16(
            q: *const std::ffi::c_void,
            k: *const std::ffi::c_void,
            v: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_qo_heads: i32,
            num_kv_heads: i32,
            seq_len: i32,
            head_dim: i32,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        /// Batch decode with paged KV cache (fp16).
        pub fn flashinfer_batch_decode_paged_f16(
            q: *const std::ffi::c_void,
            k_data: *const std::ffi::c_void,
            v_data: *const std::ffi::c_void,
            kv_indices: *const i32,
            kv_indptr: *const i32,
            kv_last_page_len: *const i32,
            output: *mut std::ffi::c_void,
            batch_size: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            page_size: i32,
            head_dim: i32,
            stream: *mut std::ffi::c_void,
        ) -> i32;
    }
}
