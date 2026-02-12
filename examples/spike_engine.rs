//! gpu-infer spike: GenerationEngine — the clean multi-sequence API.
//!
//! Proves that the engine handles prefill, decode, sampling, and lifecycle
//! automatically. You just submit() prompts and step() until done.

#[cfg(feature = "llama-cpp")]
fn main() {
    use gpu_infer::backend::llama_cpp::LlamaCppBackend;
    use gpu_infer::engine::{EngineConfig, GenerationEngine, SamplingConfig};
    use std::io::Write;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: spike_engine <model.gguf> [max_tokens]");
        std::process::exit(1);
    });

    let max_tokens: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    println!("gpu-infer spike: GenerationEngine");
    println!();

    // Load model
    print!("Loading model... ");
    std::io::stdout().flush().unwrap();
    let t0 = std::time::Instant::now();
    let backend = LlamaCppBackend::load_gguf(&model_path, u32::MAX, 4096, 8)
        .expect("Failed to load model");
    println!("OK ({:.1}s)", t0.elapsed().as_secs_f64());

    let config = backend.model_config();
    println!("  Model: {}", config.name);
    println!("  Architecture: {} layers, {} heads", config.num_layers, config.num_qo_heads);
    println!();

    // Create engine
    let mut engine = GenerationEngine::new(
        backend,
        EngineConfig {
            sampling: SamplingConfig {
                temperature: 0.7,
                top_p: 0.9,
                seed: 42,
            },
            max_batch_size: 8,
        },
    );

    // Submit three prompts — they'll all generate concurrently
    let prompts = [
        "The meaning of life is",
        "In Rust, ownership means",
        "Once upon a time, in a galaxy",
    ];

    println!("Submitting {} sequences:", prompts.len());
    let mut seq_ids = Vec::new();
    for prompt in &prompts {
        let id = engine.submit(prompt, max_tokens).expect("submit failed");
        println!("  Seq {:?}: \"{}\"", id, prompt);
        seq_ids.push(id);
    }
    println!();

    // Generation loop — step until all sequences are done
    println!("Generating (streaming):");
    println!();
    let t0 = std::time::Instant::now();
    let mut total_tokens = 0u32;
    let mut step_count = 0u32;

    while engine.has_active() {
        let new_tokens = engine.step().expect("step failed");

        for (id, token) in &new_tokens {
            if let Ok(piece) = engine.detokenize(&[*token]) {
                // Print with sequence label on first token
                if engine.get_output(*id).map(|o| o.len()) == Some(1) {
                    print!("\n  [Seq {}] ", id.0);
                }
                // Print inline — newlines get indented
                let piece = piece.replace('\n', "\n          ");
                print!("{}", piece);
                std::io::stdout().flush().unwrap();
            }
            total_tokens += 1;
        }

        step_count += 1;
    }

    let elapsed_ms = t0.elapsed().as_millis();
    let tok_per_sec = if elapsed_ms > 0 {
        total_tokens as f64 / (elapsed_ms as f64 / 1000.0)
    } else {
        0.0
    };

    // Print final outputs
    println!();
    println!();
    println!("=== Final Outputs ===");
    for (i, id) in seq_ids.iter().enumerate() {
        let text = engine
            .get_text(*id)
            .ok()
            .flatten()
            .unwrap_or_else(|| "<error>".to_string());
        let tokens = engine.get_output(*id).map(|t| t.len()).unwrap_or(0);
        println!();
        println!("--- Seq {} ---", i);
        println!("  Prompt:  \"{}\"", prompts[i]);
        println!("  Output:  \"{}\"", text.trim());
        println!(
            "  Tokens:  {} ({})",
            tokens,
            if engine.is_finished(*id) {
                "finished"
            } else {
                "still active"
            }
        );
    }

    println!();
    println!("Stats:");
    println!(
        "  {} tokens across {} sequences in {} ms ({:.1} tok/s)",
        total_tokens,
        prompts.len(),
        elapsed_ms,
        tok_per_sec
    );
    println!("  {} steps", step_count);
    println!();
    println!("SPIKE PASSED: GenerationEngine works!");
}

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("This example requires the 'llama-cpp' feature.");
    eprintln!("Run with: cargo run --example spike_engine --features llama-cpp -- <model.gguf>");
    std::process::exit(1);
}
