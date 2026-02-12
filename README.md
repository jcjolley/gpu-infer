# gpu-infer

A Rust GPU inference runtime that matches vLLM throughput with 100x faster startup.

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) via the [llama-cpp-2](https://crates.io/crates/llama-cpp-2) crate, with a clean engine API for multi-sequence text generation.

## Performance

Head-to-head on RTX 4090 with Mistral 7B (Q4_K), 3 concurrent sequences, 48 max tokens:

|                    | vLLM 0.15.1 | gpu-infer |
|--------------------|-------------|-----------|
| **Throughput**     | 227 tok/s   | 333 tok/s |
| **Generation time**| 633 ms      | 366 ms    |
| **Model load**     | 111.1s      | 1.0s      |

gpu-infer is **1.5x faster** on generation throughput and **111x faster** to load. vLLM spends over a minute building merge tables, compiling CUDA graphs, and initializing its V1 engine. gpu-infer loads a GGUF and starts generating in ~1 second.

## How it works

The key optimization is **true batching** — all active sequences are forwarded through the model in a single `LlamaBatch::decode()` call per step, rather than one call per sequence. This is what vLLM does internally with PagedAttention, and it's what closes the throughput gap.

The engine handles the full generation lifecycle automatically:

```rust
use gpu_infer::backend::llama_cpp::LlamaCppBackend;
use gpu_infer::engine::{EngineConfig, GenerationEngine};

// Load any GGUF model
let backend = LlamaCppBackend::load_gguf("model.gguf", u32::MAX, 4096, 8)?;
let mut engine = GenerationEngine::new(backend, EngineConfig::default());

// Submit prompts — they generate concurrently
let s1 = engine.submit("The meaning of life is", 64)?;
let s2 = engine.submit("fn main() {", 64)?;

// Step until all sequences finish
while engine.has_active() {
    let new_tokens = engine.step()?;
    for (id, token) in &new_tokens {
        print!("{}", engine.detokenize(&[*token])?);
    }
}

let text = engine.get_text(s1)?;
```

## Architecture

```
src/
├── engine.rs          # GenerationEngine — submit/step/get_output API
├── backend/
│   ├── mod.rs         # ModelBackend trait
│   └── llama_cpp.rs   # llama.cpp backend (GGUF loading, batched forward, sampling)
├── pool/
│   ├── mod.rs         # VRAM pool with page-granularity allocation
│   └── kv_cache.rs    # Paged KV cache manager
├── scheduler.rs       # Multi-sequence lifecycle scheduler
├── attention/
│   └── mod.rs         # FlashInfer FFI bindings (optional)
└── lib.rs
```

### Engine

`GenerationEngine` manages the full lifecycle: submit prompts, automatic prefill, batched decode, per-sequence sampling, EOS detection. Each sequence gets its own sampler with a unique seed.

### Backend

The `LlamaCppBackend` wraps llama-cpp-2 with three forward modes:
- `forward()` — single sequence (prefill or decode)
- `forward_batch()` — multiple sequences in one decode call (the hot path)
- `forward_prefill_batch()` — multiple sequences' prompts prefilled in one call

### Pool & KV Cache

Page-granularity VRAM allocation via `cudarc`. The paged KV cache tracks per-sequence page tables, allocates on demand, and frees on sequence completion. Designed for multi-model scenarios where multiple models share one GPU.

## Building

Requires CUDA and a GPU with compute capability 8.0+ (RTX 3000 series or newer).

```bash
# Clone
git clone https://github.com/jcjolley/gpu-infer.git
cd gpu-infer

# Build (defaults to llama-cpp feature)
cargo build --release

# Run the engine example with any GGUF model
cargo run --release --example spike_engine -- /path/to/model.gguf 48
```

### Features

- `llama-cpp` (default) — llama.cpp backend for GGUF model inference
- `flashinfer` — FlashInfer attention kernels (experimental)

## Examples

| Example | What it proves |
|---------|---------------|
| `spike_single_decode` | Single-token decode through FlashInfer attention |
| `spike_batch_decode` | Batched decode with paged KV cache |
| `spike_generate` | Single-sequence text generation end-to-end |
| `spike_multi_generate` | Multiple sequences generating concurrently |
| `spike_engine` | Full GenerationEngine API with streaming output |

Run any example:
```bash
cargo run --release --example spike_engine -- /path/to/model.gguf
```

## Status

This is a working prototype — the engine API is functional and fast, but it's early. Things that exist:

- Multi-sequence generation with true batching
- Automatic prefill and decode lifecycle
- Per-sequence sampling (temperature, top-p, seeded)
- VRAM pool with page-granularity allocation
- Paged KV cache

Things that don't exist yet:

- Continuous batching (new sequences joining mid-generation)
- CUDA graph capture
- Speculative decoding
- Streaming API beyond step-by-step
- Model-agnostic backend (currently llama.cpp only)

## License

MIT
