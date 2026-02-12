# gpu-infer Spike Plan

**Goal:** Prove we can call FlashInfer's paged attention CUDA kernel from Rust via FFI on the RTX 4090.

**Not a goal (yet):** Model loading, inference pipelines, multi-model orchestration. That's the real project — this spike just proves the build chain works.

## What We're Testing

One operation: **paged KV-cache decode attention**. This is the core primitive everything else builds on.

The call chain:
```
Rust main() → FFI → C++ wrapper → FlashInfer template kernel → GPU
```

## Architecture

```
gpu-infer/
├── Cargo.toml
├── build.rs              # Compiles C++ wrapper via cc crate + nvcc
├── kernels/
│   ├── wrapper.cu        # Thin C wrapper around FlashInfer's templated API
│   └── wrapper.h         # C-compatible function declarations
├── src/
│   ├── main.rs           # Spike entry point — allocate, call, verify
│   └── ffi.rs            # extern "C" declarations matching wrapper.h
└── third_party/
    └── flashinfer/       # git submodule → flashinfer-ai/flashinfer
```

## The 5 Steps

### Step 1: Project scaffold + FlashInfer submodule

```bash
cargo init gpu-infer
cd gpu-infer
git submodule add https://github.com/flashinfer-ai/flashinfer.git third_party/flashinfer
```

Dependencies:
- `cc` crate (build.rs) — compiles CUDA via nvcc
- `cuda-runtime-sys` or raw `libcudart` linking — for `cudaMalloc`, `cudaMemcpy`, `cudaStream`

### Step 2: C++ wrapper (`kernels/wrapper.cu`)

FlashInfer's API is heavily templated C++ — can't call templates directly from Rust FFI. We write a thin `extern "C"` wrapper that instantiates the template with concrete types:

```cpp
#include "flashinfer/attention/decode.cuh"

extern "C" {

// Concrete instantiation: fp16 Q/K/V, head_dim=128, no positional encoding
int flashinfer_paged_decode_f16(
    // Query: [num_queries, num_heads, head_dim]
    const void* q,
    // Paged KV cache
    const void* kv_data,        // [num_pages, 2, page_size, num_kv_heads, head_dim]
    const int32_t* kv_indices,  // page table: [num_seqs, max_pages_per_seq]
    const int32_t* kv_indptr,   // CSR-style offsets into kv_indices
    const int32_t* kv_last_page_len,  // tokens in last page per seq
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,       // for GQA: num_heads / num_kv_heads = group size
    int32_t page_size,
    int32_t head_dim,
    // Output: [num_queries, num_heads, head_dim]
    void* output,
    // CUDA stream
    void* stream
);

}
```

The implementation instantiates FlashInfer's `BatchDecodeWithPagedKVCacheDispatched<128, PosEncodingMode::kNone, ...>` with the right types.

### Step 3: build.rs

```rust
fn main() {
    // Find CUDA
    let cuda_path = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/opt/cuda".to_string());

    cc::Build::new()
        .cuda(true)
        .file("kernels/wrapper.cu")
        .include("third_party/flashinfer/include")
        .include(format!("{}/include", cuda_path))
        .flag("-gencode=arch=compute_89,code=sm_89")  // RTX 4090
        .flag("-std=c++17")
        .flag("-O2")
        .compile("flashinfer_wrapper");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
```

### Step 4: Rust FFI + GPU memory helpers (`src/ffi.rs`, `src/main.rs`)

```rust
// ffi.rs
extern "C" {
    pub fn flashinfer_paged_decode_f16(
        q: *const std::ffi::c_void,
        kv_data: *const std::ffi::c_void,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        kv_last_page_len: *const i32,
        num_seqs: i32,
        num_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        output: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
```

Main allocates dummy data on GPU, calls the kernel, copies result back, verifies shape:

```rust
fn main() {
    // Config matching a small model
    let num_seqs = 1;
    let num_heads = 32;
    let num_kv_heads = 8;   // GQA ratio 4:1
    let head_dim = 128;
    let page_size = 16;
    let num_pages = 4;      // 64 tokens of context
    let seq_len = 50;       // 50 tokens in the sequence

    // Allocate GPU memory for:
    // - query: [1, 32, 128] fp16
    // - kv_data: [4, 2, 16, 8, 128] fp16
    // - page table: [1, 4] i32
    // - output: [1, 32, 128] fp16

    // Fill query with ones, KV cache with random/ones
    // Call flashinfer_paged_decode_f16
    // Copy output back to CPU
    // Verify: output is [1, 32, 128] and non-zero

    println!("Paged decode attention: OK");
    println!("  Sequences: {}", num_seqs);
    println!("  Heads: {} Q, {} KV (GQA {}:1)", num_heads, num_kv_heads, num_heads/num_kv_heads);
    println!("  Head dim: {}", head_dim);
    println!("  Pages: {} x {} tokens = {} capacity", num_pages, page_size, num_pages * page_size);
    println!("  Sequence length: {}", seq_len);
}
```

### Step 5: Run it

```bash
cargo run --release
```

## Success Criteria

1. `build.rs` compiles FlashInfer headers + our wrapper via nvcc without errors
2. Rust binary links against CUDA runtime and our wrapper
3. Kernel executes on GPU and returns non-zero output of correct shape `[1, 32, 128]`
4. No GPU errors (checked via `cudaGetLastError`)

## Known Risks

| Risk | Mitigation |
|------|------------|
| FlashInfer templates require specific CUDA features | We have CUDA 13.1, compute 8.9 — should be fine. FlashInfer supports CUDA 12.6+. |
| Template instantiation pulls in massive header graph | Build time may be long. Limit to one concrete instantiation. |
| FlashInfer's internal `Params` struct is complex | Start with the simplest variant. May need to dig into their source to understand the exact struct layout. |
| `cc` crate CUDA support may have quirks | Fallback: raw `nvcc` command in build.rs via `Command::new("nvcc")` |
| FlashInfer might require CuTe/CUTLASS headers | It bundles these as submodules. May need to init `third_party/flashinfer`'s own submodules. |
| Half-precision (`__half`) type handling across FFI | Use `void*` pointers in the C API, cast inside the wrapper. Rust side treats them as opaque byte buffers. |

## What Comes After (if spike succeeds)

The spike proves: "Rust can call FlashInfer's attention kernels on our GPU." From there:

1. **CUDA memory pool** — `cudaMallocAsync`/`cudaFreeAsync` with stream-ordered allocation
2. **Page table manager** — Rust struct that tracks allocated/free pages across multiple models
3. **Prefix cache** — Hash system prompt → cached KV pages, skip recomputation
4. **Model integration** — Wire into candle (Voxtral) and llama-cpp-rs (polish model) as custom attention backends
5. **Speculative decoding** — Draft model on same GPU, shared page pool

## Environment

- **GPU:** NVIDIA GeForce RTX 4090 (24GB, compute 8.9)
- **CUDA:** 13.1 (V13.1.115)
- **Driver:** 590.48.01
- **OS:** Linux (CachyOS)
- **Rust:** stable

## Time Estimate

Not giving one. Could be an afternoon if the templates cooperate, could be a week if we're fighting CUDA linker errors. That's what spikes are for.
