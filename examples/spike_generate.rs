//! gpu-infer spike: End-to-end text generation via llama.cpp backend.
//!
//! Loads a GGUF model, tokenizes a prompt, generates tokens one at a time,
//! and prints the output. This proves the full pipeline:
//! LlamaCppBackend → tokenize → prefill → decode loop → detokenize → text

#[cfg(feature = "llama-cpp")]
fn main() {
    use gpu_infer::backend::llama_cpp::LlamaCppBackend;
    use llama_cpp_2::sampling::LlamaSampler;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: spike_generate <model.gguf> [prompt]");
        eprintln!("Example: spike_generate /path/to/model.gguf \"Hello, world!\"");
        std::process::exit(1);
    });

    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The quick brown fox".to_string());

    let max_tokens: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    println!("gpu-infer spike: Text generation via llama.cpp backend");
    println!("  Model: {}", model_path);
    println!("  Prompt: \"{}\"", prompt);
    println!("  Max tokens: {}", max_tokens);
    println!();

    // Load model — full GPU offload
    print!("Loading model... ");
    let t0 = std::time::Instant::now();
    let mut backend = LlamaCppBackend::load_gguf(&model_path, u32::MAX, 2048, 1)
        .expect("Failed to load model");
    println!("OK ({:.1}s)", t0.elapsed().as_secs_f64());

    let config = backend.model_config();
    println!("  Architecture: {} layers, {} QO heads, {} KV heads, dim {}",
        config.num_layers, config.num_qo_heads, config.num_kv_heads, config.head_dim);
    println!("  Vocab: {} tokens", config.vocab_size);
    println!("  EOS token: {}", backend.eos_token());
    println!();

    // Tokenize prompt
    let prompt_tokens = backend.tokenize_str(&prompt, true).expect("tokenize failed");
    println!("Prompt tokens ({}): {:?}", prompt_tokens.len(), &prompt_tokens[..prompt_tokens.len().min(20)]);
    println!();

    // Prefill — feed all prompt tokens at once
    print!("Prefill ({} tokens)... ", prompt_tokens.len());
    let t0 = std::time::Instant::now();
    let _logits = backend.forward(&prompt_tokens, 0, 0).expect("prefill failed");
    let prefill_ms = t0.elapsed().as_millis();
    println!("OK ({} ms, {:.0} tok/s)",
        prefill_ms,
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0));

    // Create sampler
    let mut sampler = LlamaCppBackend::make_sampler(0.7, 0.9, 42);

    // Sample first token from prefill logits
    let batch_idx = (prompt_tokens.len() - 1) as i32;
    let mut next_token = backend.sample_next(&mut sampler, batch_idx);

    // Decode loop
    print!("\nGenerated: ");
    let eos = backend.eos_token();
    let mut generated = Vec::new();
    let t0 = std::time::Instant::now();

    for step in 0..max_tokens {
        if next_token == eos {
            break;
        }

        generated.push(next_token);

        // Print token as text
        if let Ok(piece) = backend.detokenize_tokens(&[next_token]) {
            print!("{}", piece);
        }

        // Forward one token
        let pos = (prompt_tokens.len() + step as usize) as i32;
        let _logits = backend.forward(&[next_token], 0, pos).expect("decode failed");

        // Sample next
        next_token = backend.sample_next(&mut sampler, 0);
    }

    let decode_ms = t0.elapsed().as_millis();
    let tok_per_sec = if decode_ms > 0 {
        generated.len() as f64 / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };

    println!();
    println!();
    println!("Stats:");
    println!("  Generated: {} tokens in {} ms ({:.1} tok/s)",
        generated.len(), decode_ms, tok_per_sec);
    println!("  Stopped: {}",
        if next_token == eos { "EOS" } else { "max_tokens" });
    println!();
    println!("SPIKE PASSED: Full text generation pipeline works!");
}

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("This example requires the 'llama-cpp' feature.");
    eprintln!("Run with: cargo run --example spike_generate --features llama-cpp -- <model.gguf>");
    std::process::exit(1);
}
