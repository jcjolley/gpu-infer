//! gpu-infer spike: Multi-sequence batched generation via llama.cpp.
//!
//! Proves that multiple prompts can generate concurrently through the same
//! model. This is the foundation for the self-feeding loop: voice
//! transcription + polish running in parallel on one GPU.
//!
//! llama.cpp handles this via seq_id in LlamaBatch — each token belongs to
//! a sequence, and the KV cache is keyed per sequence.

#[cfg(feature = "llama-cpp")]
fn main() {
    use gpu_infer::backend::llama_cpp::LlamaCppBackend;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: spike_multi_generate <model.gguf> [max_tokens]");
        std::process::exit(1);
    });

    let max_tokens: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    // Two prompts that should produce very different continuations
    let prompts = [
        "The capital of France is",
        "fn main() { println!(",
    ];
    let n_seqs = prompts.len();

    println!("gpu-infer spike: Multi-sequence batched generation");
    println!("  Model: {}", model_path);
    println!("  Sequences: {}", n_seqs);
    println!("  Max tokens per sequence: {}", max_tokens);
    println!();

    // Load model with n_seq_max = number of concurrent sequences
    print!("Loading model... ");
    let t0 = std::time::Instant::now();
    let mut backend =
        LlamaCppBackend::load_gguf(&model_path, u32::MAX, 4096, n_seqs as u32)
            .expect("Failed to load model");
    println!("OK ({:.1}s)", t0.elapsed().as_secs_f64());

    let eos = backend.eos_token();

    // Tokenize all prompts
    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let tokens = backend.tokenize_str(prompt, true).expect("tokenize failed");
            println!("  Seq {}: \"{}\" → {} tokens", i, prompt, tokens.len());
            tokens
        })
        .collect();
    println!();

    // Prefill each sequence and sample first token immediately.
    // llama.cpp overwrites logits on each decode, so we must sample
    // right after each sequence's prefill.
    let mut samplers: Vec<_> = (0..n_seqs)
        .map(|i| LlamaCppBackend::make_sampler(0.7, 0.9, 42 + i as u32))
        .collect();

    let mut next_tokens = Vec::new();
    let mut positions: Vec<i32> = Vec::new();

    println!("Prefill + first sample:");
    let t0 = std::time::Instant::now();
    for (seq_id, tokens) in prompt_tokens.iter().enumerate() {
        let _logits = backend
            .forward(tokens, seq_id as i32, 0)
            .expect("prefill failed");

        let batch_idx = (tokens.len() - 1) as i32;
        let first_tok = backend.sample_next(&mut samplers[seq_id], batch_idx);
        next_tokens.push(first_tok);
        positions.push(tokens.len() as i32);

        if let Ok(piece) = backend.detokenize_tokens(&[first_tok]) {
            println!(
                "  Seq {}: first token = \"{}\" ({})",
                seq_id,
                piece.trim(),
                first_tok
            );
        }
    }
    let prefill_ms = t0.elapsed().as_millis();
    let total_prefill: usize = prompt_tokens.iter().map(|t| t.len()).sum();
    println!(
        "  Prefill: {} tokens in {} ms ({:.0} tok/s)",
        total_prefill,
        prefill_ms,
        total_prefill as f64 / (prefill_ms as f64 / 1000.0)
    );
    println!();

    // Decode loop — generate for all active sequences each step
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); n_seqs];
    let mut finished = vec![false; n_seqs];

    println!("Decode loop:");
    let t0 = std::time::Instant::now();
    let mut total_steps = 0u32;

    for _step in 0..max_tokens {
        // Record tokens and check EOS
        for i in 0..n_seqs {
            if finished[i] {
                continue;
            }
            generated[i].push(next_tokens[i]);
            if next_tokens[i] == eos {
                finished[i] = true;
            }
        }

        let active: Vec<usize> = (0..n_seqs).filter(|&i| !finished[i]).collect();
        if active.is_empty() {
            break;
        }

        // Forward each active sequence's next token
        for &i in &active {
            let _logits = backend
                .forward(&[next_tokens[i]], i as i32, positions[i])
                .expect("decode failed");
            positions[i] += 1;

            next_tokens[i] = backend.sample_next(&mut samplers[i], 0);
        }

        total_steps += 1;
    }

    let decode_ms = t0.elapsed().as_millis();
    let total_tokens: usize = generated.iter().map(|g| g.len()).sum();
    let tok_per_sec = if decode_ms > 0 {
        total_tokens as f64 / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };

    // Print results
    println!();
    for (i, prompt) in prompts.iter().enumerate() {
        let text = backend
            .detokenize_tokens(&generated[i])
            .unwrap_or_else(|_| "<detokenize error>".to_string());
        println!("--- Seq {} ---", i);
        println!("  Prompt: \"{}\"", prompt);
        println!("  Output: \"{}\"", text.trim());
        println!(
            "  Tokens: {} (stopped: {})",
            generated[i].len(),
            if generated[i].last() == Some(&eos) {
                "EOS"
            } else {
                "max_tokens"
            }
        );
    }

    println!();
    println!("Stats:");
    println!(
        "  Total: {} tokens across {} sequences in {} ms ({:.1} tok/s)",
        total_tokens, n_seqs, decode_ms, tok_per_sec
    );
    println!("  Steps: {}", total_steps);
    println!();
    println!("SPIKE PASSED: Multi-sequence generation works!");
}

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("This example requires the 'llama-cpp' feature.");
    eprintln!(
        "Run with: cargo run --example spike_multi_generate --features llama-cpp -- <model.gguf>"
    );
    std::process::exit(1);
}
