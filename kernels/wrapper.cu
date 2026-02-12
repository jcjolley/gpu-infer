// C wrappers around FlashInfer's templated attention kernels.
// Instantiates for fp16, head_dim=128, no positional encoding, standard softmax attention.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/page.cuh"

#include "wrapper.h"

using namespace flashinfer;

// Standard attention: no mask, no sliding window, no logits cap, no alibi
using StdAttention = DefaultAttention</*use_custom_mask=*/false,
                                       /*use_sliding_window=*/false,
                                       /*use_logits_soft_cap=*/false,
                                       /*use_alibi=*/false>;

// ============================================================================
// Single-sequence decode (contiguous KV cache)
// ============================================================================

extern "C" int flashinfer_single_decode_f16(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int num_qo_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    void* stream
) {
    using DType = half;
    using Params = SingleDecodeParams<DType, DType, DType>;

    Params params(
        const_cast<DType*>(static_cast<const DType*>(q)),
        const_cast<DType*>(static_cast<const DType*>(k)),
        const_cast<DType*>(static_cast<const DType*>(v)),
        static_cast<DType*>(output),
        /*maybe_alibi_slopes=*/nullptr,
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        QKVLayout::kNHD,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        /*sm_scale=*/1.0f / sqrtf(static_cast<float>(head_dim)),
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e4f
    );

    cudaError_t err = SingleDecodeWithKVCacheDispatched<
        128,
        PosEncodingMode::kNone,
        StdAttention,
        Params
    >(params, /*tmp=*/nullptr, static_cast<cudaStream_t>(stream));

    return static_cast<int>(err);
}

// ============================================================================
// Batch decode with paged KV cache
// ============================================================================

extern "C" int flashinfer_batch_decode_paged_f16(
    // Query: [batch_size, num_qo_heads, head_dim] fp16
    const void* q,
    // Paged KV cache data (K and V stored separately)
    // k_data: [max_num_pages, num_kv_heads, page_size, head_dim] fp16 (HND layout)
    const void* k_data,
    // v_data: [max_num_pages, num_kv_heads, page_size, head_dim] fp16 (HND layout)
    const void* v_data,
    // Page table: indices into k_data/v_data pages
    // [sum of pages across all sequences] (CSR format with indptr)
    const int32_t* kv_indices,
    // CSR indptr: [batch_size + 1], indptr[0]=0, indptr[batch_size]=total_pages
    const int32_t* kv_indptr,
    // Tokens in the last page of each sequence: [batch_size]
    const int32_t* kv_last_page_len,
    // Output: [batch_size, num_qo_heads, head_dim] fp16
    void* output,
    // Dimensions
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    // CUDA stream
    void* stream
) {
    using DType = half;
    using IdType = int32_t;
    using Params = BatchDecodeParams<DType, DType, DType, IdType>;

    // Build paged_kv_t descriptor
    paged_kv_t<DType, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(batch_size),
        QKVLayout::kHND,
        const_cast<DType*>(static_cast<const DType*>(k_data)),
        const_cast<DType*>(static_cast<const DType*>(v_data)),
        const_cast<IdType*>(kv_indices),
        const_cast<IdType*>(kv_indptr),
        const_cast<IdType*>(kv_last_page_len),
        /*rope_pos_offset=*/nullptr
    );

    IdType q_stride_n = num_qo_heads * head_dim;
    IdType q_stride_h = head_dim;

    Params params(
        const_cast<DType*>(static_cast<const DType*>(q)),
        /*q_rope_offset=*/nullptr,
        paged_kv,
        static_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        static_cast<uint32_t>(num_qo_heads),
        q_stride_n,
        q_stride_h,
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        /*sm_scale=*/1.0f / sqrtf(static_cast<float>(head_dim)),
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e4f
    );

    // Non-partitioned decode still needs scheduling arrays (kernel reads unconditionally).
    // request_indices: identity mapping [0, 1, ..., batch_size-1]
    // kv_tile_indices: all zeros (no KV tiling)
    // kv_chunk_size_ptr: points to a dummy value (read but unused when !partition_kv)
    IdType* d_request_indices = nullptr;
    IdType* d_kv_tile_indices = nullptr;
    IdType* d_kv_chunk_size = nullptr;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaMalloc(&d_request_indices, batch_size * sizeof(IdType));
    cudaMalloc(&d_kv_tile_indices, batch_size * sizeof(IdType));
    cudaMalloc(&d_kv_chunk_size, sizeof(IdType));

    // Fill request_indices with identity [0, 1, 2, ...]
    {
        std::vector<IdType> h_request(batch_size);
        std::vector<IdType> h_tile(batch_size, 0);
        IdType h_chunk = 0;
        for (int i = 0; i < batch_size; i++) h_request[i] = i;
        cudaMemcpy(d_request_indices, h_request.data(), batch_size * sizeof(IdType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kv_tile_indices, h_tile.data(), batch_size * sizeof(IdType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kv_chunk_size, &h_chunk, sizeof(IdType), cudaMemcpyHostToDevice);
    }

    params.padded_batch_size = batch_size;
    params.partition_kv = false;
    params.request_indices = d_request_indices;
    params.kv_tile_indices = d_kv_tile_indices;
    params.kv_chunk_size_ptr = d_kv_chunk_size;
    params.block_valid_mask = nullptr;

    cudaError_t err = BatchDecodeWithPagedKVCacheDispatched<
        128,
        PosEncodingMode::kNone,
        StdAttention,
        Params
    >(params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false,
      cuda_stream);

    // Sync before freeing temp buffers
    cudaStreamSynchronize(cuda_stream);
    cudaFree(d_request_indices);
    cudaFree(d_kv_tile_indices);
    cudaFree(d_kv_chunk_size);

    return static_cast<int>(err);
}
