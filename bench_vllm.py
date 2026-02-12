"""Head-to-head benchmark: vLLM vs gpu-infer.

Same model (Mistral 7B from Ollama blob), same prompts, same max tokens.
"""

import time
from vllm import LLM, SamplingParams

MODEL_PATH = "/var/lib/ollama/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"

PROMPTS = [
    "The meaning of life is",
    "In Rust, ownership means",
    "Once upon a time, in a galaxy",
]

MAX_TOKENS = 48

print("vLLM benchmark: Mistral 7B, 3 sequences")
print(f"  Max tokens per sequence: {MAX_TOKENS}")
print()

# Load model
print("Loading model...")
t0 = time.time()
llm = LLM(model=MODEL_PATH, max_model_len=4096, gpu_memory_utilization=0.85)
load_time = time.time() - t0
print(f"  Loaded in {load_time:.1f}s")
print()

# Generate — vLLM batches these internally
sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=MAX_TOKENS, seed=42)

print("Generating...")
t0 = time.time()
outputs = llm.generate(PROMPTS, sampling)
gen_time = time.time() - t0

# Results
total_tokens = 0
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    n_tokens = len(output.outputs[0].token_ids)
    total_tokens += n_tokens
    print(f"\n--- Seq {i} ---")
    print(f"  Prompt: \"{PROMPTS[i]}\"")
    print(f"  Output: \"{text.strip()[:200]}\"")
    print(f"  Tokens: {n_tokens}")

tok_per_sec = total_tokens / gen_time if gen_time > 0 else 0

print()
print("Stats:")
print(f"  {total_tokens} tokens across {len(PROMPTS)} sequences in {gen_time*1000:.0f} ms ({tok_per_sec:.1f} tok/s)")
print(f"  Model load: {load_time:.1f}s")
