//! gpu-infer spike: Continuous batching — submit new prompts mid-generation.
//!
//! Proves that the engine handles late arrivals: start generating seq 0,
//! then inject seq 1 and seq 2 after N steps. All three should complete.

#[cfg(feature = "llama-cpp")]
fn main() {
    use gpu_infer::backend::llama_cpp::LlamaCppBackend;
    use gpu_infer::engine::{EngineConfig, GenerationEngine, SamplingConfig};
    use std::io::Write;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: spike_continuous_batch <model.gguf> [max_tokens]");
        std::process::exit(1);
    });

    let max_tokens: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    println!("gpu-infer spike: Continuous Batching");
    println!();

    // Load model
    print!("Loading model... ");
    std::io::stdout().flush().unwrap();
    let t0 = std::time::Instant::now();
    let backend = LlamaCppBackend::load_gguf(&model_path, u32::MAX, 4096, 8)
        .expect("Failed to load model");
    println!("OK ({:.1}s)", t0.elapsed().as_secs_f64());
    println!();

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

    // Phase 1: Submit first prompt and start generating
    println!("=== Phase 1: Submit seq 0, generate 10 steps ===");
    let s0 = engine.submit("The meaning of life is", max_tokens).expect("submit failed");
    println!("  Submitted seq {:?}: \"The meaning of life is\"", s0);
    println!();

    let t0 = std::time::Instant::now();
    let mut total_tokens = 0u32;

    for step in 0..10 {
        let new_tokens = engine.step().expect("step failed");
        for (id, token) in &new_tokens {
            if let Ok(piece) = engine.detokenize(&[*token]) {
                if engine.get_output(*id).map(|o| o.len()) == Some(1) {
                    print!("\n  [Seq {}] ", id.0);
                }
                print!("{}", piece.replace('\n', "\n          "));
                std::io::stdout().flush().unwrap();
            }
            total_tokens += 1;
        }

        if step == 9 {
            println!();
            println!();
            println!("  (10 steps done, seq 0 has {} tokens so far)",
                engine.get_output(s0).map(|o| o.len()).unwrap_or(0));
        }
    }

    // Phase 2: Inject two more prompts mid-generation
    println!();
    println!("=== Phase 2: Submit seq 1 and seq 2 MID-GENERATION ===");
    let s1 = engine.submit("In Rust, ownership means", max_tokens).expect("submit failed");
    println!("  Submitted seq {:?}: \"In Rust, ownership means\"", s1);
    let s2 = engine.submit("Once upon a time, in a galaxy", max_tokens).expect("submit failed");
    println!("  Submitted seq {:?}: \"Once upon a time, in a galaxy\"", s2);
    println!();
    println!("  Active sequences: {} (seq 0 still generating + 2 new)", engine.num_active());
    println!();

    // Phase 3: Continue stepping until all done
    println!("=== Phase 3: Continue until all sequences complete ===");
    let mut step_count = 10u32;

    while engine.has_active() {
        let new_tokens = engine.step().expect("step failed");
        for (id, token) in &new_tokens {
            if let Ok(piece) = engine.detokenize(&[*token]) {
                if engine.get_output(*id).map(|o| o.len()) == Some(1) {
                    print!("\n  [Seq {}] ", id.0);
                }
                print!("{}", piece.replace('\n', "\n          "));
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

    // Final outputs
    println!();
    println!();
    println!("=== Final Outputs ===");
    for (i, id) in [s0, s1, s2].iter().enumerate() {
        let text = engine
            .get_text(*id)
            .ok()
            .flatten()
            .unwrap_or_else(|| "<error>".to_string());
        let tokens = engine.get_output(*id).map(|t| t.len()).unwrap_or(0);
        let prompts = [
            "The meaning of life is",
            "In Rust, ownership means",
            "Once upon a time, in a galaxy",
        ];
        println!();
        println!("--- Seq {} ---", i);
        println!("  Prompt:  \"{}\"", prompts[i]);
        println!("  Output:  \"{}\"", text.trim());
        println!(
            "  Tokens:  {} ({})",
            tokens,
            if engine.is_finished(*id) { "finished" } else { "still active" }
        );
    }

    println!();
    println!("Stats:");
    println!(
        "  {} tokens across 3 sequences in {} ms ({:.1} tok/s)",
        total_tokens, elapsed_ms, tok_per_sec
    );
    println!("  {} total steps", step_count);
    println!("  Seq 0 started at step 0, seqs 1-2 joined at step 10");
    println!();
    println!("SPIKE PASSED: Continuous batching works!");
}

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("This example requires the 'llama-cpp' feature.");
    eprintln!("Run with: cargo run --example spike_continuous_batch --features llama-cpp -- <model.gguf>");
    std::process::exit(1);
}
