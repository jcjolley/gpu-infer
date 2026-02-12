#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single-sequence decode attention with contiguous KV cache.
// q:      [num_qo_heads, head_dim] fp16
// k:      [seq_len, num_kv_heads, head_dim] fp16 (NHD layout)
// v:      [seq_len, num_kv_heads, head_dim] fp16 (NHD layout)
// output: [num_qo_heads, head_dim] fp16
// Returns 0 on success, non-zero CUDA error code on failure.
int flashinfer_single_decode_f16(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int num_qo_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    void* stream
);

// Batch decode with paged KV cache.
// q:              [batch_size, num_qo_heads, head_dim] fp16
// k_data/v_data:  [max_num_pages, num_kv_heads, page_size, head_dim] fp16 (HND)
// kv_indices:     [total_pages] page indices (CSR values)
// kv_indptr:      [batch_size + 1] CSR offsets
// kv_last_page_len: [batch_size] tokens in last page per sequence
// output:         [batch_size, num_qo_heads, head_dim] fp16
// Returns 0 on success, non-zero CUDA error code on failure.
int flashinfer_batch_decode_paged_f16(
    const void* q,
    const void* k_data,
    const void* v_data,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* kv_last_page_len,
    void* output,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    void* stream
);

#ifdef __cplusplus
}
#endif
