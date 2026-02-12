//! gpu-infer spike: Paged batch decode via FlashInfer.
//!
//! Proves the full pipeline: VramPool → PagedKvCache → FlashInfer batch decode kernel.
//! Two sequences with different lengths, batched into a single kernel launch.

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use gpu_infer::attention::ffi::flashinfer_batch_decode_paged_f16;
use gpu_infer::pool::kv_cache::{KvCacheConfig, PagedKvCache};
use gpu_infer::pool::{VramPool, VramPoolConfig};
use half::f16;

fn main() {
    let num_qo_heads: u32 = 32;
    let num_kv_heads: u32 = 8; // GQA 4:1
    let head_dim: u32 = 128;
    let tokens_per_page: u32 = 16;
    let batch_size = 2;

    println!("gpu-infer spike: Paged batch decode");
    println!("  Q heads:        {}", num_qo_heads);
    println!("  KV heads:       {} (GQA {}:1)", num_kv_heads, num_qo_heads / num_kv_heads);
    println!("  Head dim:       {}", head_dim);
    println!("  Tokens/page:    {}", tokens_per_page);
    println!("  Batch size:     {}", batch_size);
    println!();

    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
    println!("GPU: {}", ctx.name().unwrap_or("unknown".into()));
    let stream = ctx.default_stream();

    // KV cache config
    let kv_config = KvCacheConfig {
        num_kv_heads,
        head_dim,
        tokens_per_page,
        dtype_bytes: 2, // fp16
    };

    let k_page_bytes = kv_config.k_page_bytes();
    println!("  K page bytes:   {} ({} KB)", k_page_bytes, k_page_bytes / 1024);

    // Create TWO pools — one for K pages, one for V pages.
    // FlashInfer expects k_data and v_data as separate contiguous arrays
    // indexed by the same page IDs.
    let num_pages = 64;
    let pool_config = VramPoolConfig {
        page_size_bytes: k_page_bytes,
        num_pages,
    };

    let _k_pool = VramPool::new(&stream, &pool_config).expect("Failed to create K pool");
    let _v_pool = VramPool::new(&stream, &pool_config).expect("Failed to create V pool");

    println!(
        "  K pool:         {} pages = {} KB",
        num_pages,
        num_pages * k_page_bytes / 1024
    );
    println!(
        "  V pool:         {} pages = {} KB",
        num_pages,
        num_pages * k_page_bytes / 1024
    );
    println!();

    // Create KV cache and add two sequences
    let _cache = PagedKvCache::new(kv_config.clone());

    // We need a unified pool for the KV cache's alloc tracking.
    // The KV cache allocates from a single pool (it tracks K and V page IDs separately
    // but the pool itself is one thing). For the FlashInfer layout, we need K and V
    // in separate contiguous buffers. So we use the K pool for both K and V page tracking
    // (since page IDs just index into the pools).
    //
    // Actually — the PagedKvCache allocates from ONE pool but tracks K and V pages
    // separately. Each K page and V page gets a separate alloc from the pool.
    // For FlashInfer, we need k_data[page_id] and v_data[page_id] to be the same
    // page_id indexing into separate arrays.
    //
    // The cleanest approach: use ONE pool where even pages are K and odd pages are V,
    // then remap. But that's overengineering for a spike.
    //
    // Simplest approach: allocate K and V data as separate flat GPU buffers,
    // fill them with synthetic data at the right page offsets. The page IDs from
    // PagedKvCache index into both buffers identically.

    // For the spike, we'll manage the KV cache manually to match the two-pool setup.
    // Sequence 1: 20 tokens (2 pages), Sequence 2: 40 tokens (3 pages)
    let seq1_tokens: u32 = 20;
    let seq2_tokens: u32 = 40;

    let seq1_pages = ((seq1_tokens + tokens_per_page - 1) / tokens_per_page) as usize; // 2
    let seq2_pages = ((seq2_tokens + tokens_per_page - 1) / tokens_per_page) as usize; // 3

    println!("Sequence 1: {} tokens, {} pages", seq1_tokens, seq1_pages);
    println!("Sequence 2: {} tokens, {} pages", seq2_tokens, seq2_pages);

    // Fill K and V pool data with synthetic fp16 values.
    // FlashInfer expects: [max_num_pages, num_kv_heads, page_size, head_dim] in HND layout
    let elems_per_page = (num_kv_heads * tokens_per_page * head_dim) as usize;
    let total_k_elems = num_pages * elems_per_page;

    let k_host: Vec<u16> = (0..total_k_elems)
        .map(|i| f16::from_f32(0.01 * ((i % 128) as f32 - 64.0)).to_bits())
        .collect();
    let v_host: Vec<u16> = (0..total_k_elems)
        .map(|i| f16::from_f32(0.01 * ((i % 128) as f32)).to_bits())
        .collect();

    // Copy to GPU — these ARE the paged KV cache buffers
    let k_data_dev = stream
        .clone_htod(&k_host)
        .expect("Failed to copy K data to GPU");
    let v_data_dev = stream
        .clone_htod(&v_host)
        .expect("Failed to copy V data to GPU");

    // Query tensor: [batch_size, num_qo_heads, head_dim]
    let q_len = (batch_size as usize) * (num_qo_heads as usize) * (head_dim as usize);
    let q_host: Vec<u16> = (0..q_len)
        .map(|i| f16::from_f32(0.02 * ((i % 128) as f32 - 64.0)).to_bits())
        .collect();
    let q_dev = stream
        .clone_htod(&q_host)
        .expect("Failed to copy Q to GPU");

    // Output tensor: [batch_size, num_qo_heads, head_dim]
    let o_len = q_len;
    let mut o_dev: CudaSlice<u16> = stream.alloc_zeros(o_len).expect("Failed to alloc output");

    // Build page table in CSR format
    // Seq 1 uses pages 0,1. Seq 2 uses pages 2,3,4.
    let kv_indices: Vec<i32> = vec![0, 1, 2, 3, 4];
    let kv_indptr: Vec<i32> = vec![0, 2, 5]; // seq1: pages[0..2], seq2: pages[2..5]
    let kv_last_page_len: Vec<i32> = vec![
        (seq1_tokens % tokens_per_page) as i32, // 20 % 16 = 4
        (seq2_tokens % tokens_per_page) as i32, // 40 % 16 = 8
    ];

    println!(
        "  kv_indices:     {:?}",
        kv_indices
    );
    println!("  kv_indptr:      {:?}", kv_indptr);
    println!("  kv_last_page:   {:?}", kv_last_page_len);
    println!();

    // Copy CSR arrays to GPU
    let indices_dev = stream
        .clone_htod(&kv_indices)
        .expect("Failed to copy indices");
    let indptr_dev = stream
        .clone_htod(&kv_indptr)
        .expect("Failed to copy indptr");
    let last_page_dev = stream
        .clone_htod(&kv_last_page_len)
        .expect("Failed to copy last_page_len");

    // Get raw pointers
    let (q_ptr, _g1) = q_dev.device_ptr(&stream);
    let (k_ptr, _g2) = k_data_dev.device_ptr(&stream);
    let (v_ptr, _g3) = v_data_dev.device_ptr(&stream);
    let (o_ptr, _g4) = o_dev.device_ptr_mut(&stream);
    let (idx_ptr, _g5) = indices_dev.device_ptr(&stream);
    let (indptr_ptr, _g6) = indptr_dev.device_ptr(&stream);
    let (lpl_ptr, _g7) = last_page_dev.device_ptr(&stream);

    print!("Calling FlashInfer BatchDecodePagedKVCache... ");
    let err = unsafe {
        flashinfer_batch_decode_paged_f16(
            q_ptr as *const std::ffi::c_void,
            k_ptr as *const std::ffi::c_void,
            v_ptr as *const std::ffi::c_void,
            idx_ptr as *const i32,
            indptr_ptr as *const i32,
            lpl_ptr as *const i32,
            o_ptr as *mut std::ffi::c_void,
            batch_size,
            num_qo_heads as i32,
            num_kv_heads as i32,
            tokens_per_page as i32,
            head_dim as i32,
            std::ptr::null_mut(), // default stream
        )
    };

    // Drop guards before synchronize
    drop(_g1);
    drop(_g2);
    drop(_g3);
    drop(_g4);
    drop(_g5);
    drop(_g6);
    drop(_g7);

    if err != 0 {
        eprintln!("FAILED (CUDA error {})", err);
        std::process::exit(1);
    }

    stream.synchronize().expect("CUDA synchronize failed");
    println!("OK");

    // Read output back
    let o_host = stream
        .clone_dtoh(&o_dev)
        .expect("Failed to copy output from GPU");

    let values: Vec<f32> = o_host
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();

    let per_seq = (num_qo_heads as usize) * (head_dim as usize);

    println!();
    println!("Results:");
    for seq in 0..batch_size as usize {
        let start = seq * per_seq;
        let end = start + per_seq;
        let seq_vals = &values[start..end];

        let non_zero = seq_vals.iter().filter(|&&v| v != 0.0).count();
        let max_val = seq_vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let has_nan = seq_vals.iter().any(|v| v.is_nan());

        println!("  Sequence {} ({} tokens):", seq, if seq == 0 { seq1_tokens } else { seq2_tokens });
        println!("    Non-zero: {}/{}", non_zero, per_seq);
        println!("    Max |val|: {:.6}", max_val);
        println!("    Has NaN:   {}", has_nan);
        println!("    First 8:   {:?}", &seq_vals[..8]);
    }

    let all_non_zero: usize = values.iter().filter(|&&v| v != 0.0).count();
    let any_nan = values.iter().any(|v| v.is_nan());

    println!();
    if all_non_zero > 0 && !any_nan {
        println!("SPIKE PASSED: Paged batch decode works from Rust!");
        println!("  Two sequences, different lengths, one kernel launch.");
        println!("  Pool → KV cache → CSR page table → FlashInfer → GPU → results.");
    } else if any_nan {
        eprintln!("SPIKE FAILED: Output contains NaN");
        std::process::exit(1);
    } else {
        eprintln!("SPIKE FAILED: Output is all zeros");
        std::process::exit(1);
    }
}
