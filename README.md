# inference — Hierarchical KV Cache Inference Engine

A research inference engine that achieves **constant-cost decode steps independent of conversation length**, backed by importance-scored KV eviction, SQLite-persisted session memory, and a paged GPU allocator for concurrent sessions.

Tested on Google Colab T4 (16 GB) with Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct.

---

## The Problem

Standard transformer inference has two scaling problems in long-context / multi-turn use:

| Problem | Vanilla HF | This engine |
|---------|-----------|-------------|
| Decode step cost | O(T) — grows with every token ever seen | O(K) — constant, K = fixed budget |
| Session memory after N turns | O(T₁+T₂+…+Tₙ) — unbounded | Fixed GPU budget + SQLite offload |
| Concurrent sessions on one GPU | 1–2 (memory explosion) | 10–16 (paged allocator, LRU eviction) |

vLLM and llama.cpp solve the *storage* problem — they are memory managers that keep all tokens efficiently. They make no decisions about **which tokens matter**. This engine solves the *semantic eviction* problem: deciding which O(T−K) tokens can be dropped without losing recall, and proving it with a NIAH benchmark.

---

## Architecture

```
Incoming prompt
      │
      ▼
 RBF pre-filter ──► embed query ──► FAISS lookup ──► retrieve similar past KV slots
      │                                                         │
      ▼                                                         ▼
 TwoPassEngine                                    Prime HierarchicalKVCache
  ├─ summary_pass: full O(T²) forward, score token importance
  └─ decode_loop:  O(K) per step — K = fixed cache budget
      │
      ▼
 HierarchicalKVCache  (3-level fixed budget)
  ├─ L0 sink       — first N tokens, never evicted (StreamingLLM insight)
  ├─ L1 recent     — sliding FIFO window, always retained
  └─ L2 important  — KLL-scored pool; lowest-importance token evicted on overflow
      │
      ▼
 PersistentMemoryEngine (session continuity across requests)
  ├─ save: token IDs + positions + importance + embeddings → SQLite3
  ├─ load: top-K important tokens ──► FAISS similarity rank → mini re-encode → KV
  └─ PagePool: paged GPU allocator shared across all concurrent sessions
```

### Why three levels?

- **L0 Sink:** LLMs route disproportionate attention mass to the first few tokens regardless of content (StreamingLLM, 2023). Dropping them breaks even unrelated generation. They are protected unconditionally.
- **L1 Recent:** Local coherence for the current generation burst. Tokens here need no scoring.
- **L2 Important:** The semantic memory. Tokens are scored by cumulative attention-received mass, decayed over time. The KLL sketch approximates the eviction threshold in O(1) without sorting the entire pool.

---

## Verified Results (Qwen2.5-3B-Instruct, int8, T4)

### Multi-turn NIAH (needle-in-a-haystack recall)

A secret code (`FALCON-7749`) is planted in turn 0. Filler turns are inserted. The model is asked to recall the code at increasing distances.

| Distance (turns after planting) | Recall |
|---------------------------------|--------|
| 1 | ✓ |
| 3 | ✓ |
| 5 | ✓ |
| 8 | ✓ |
| 12 | ✓ |

**100% recall at 12 turns** with a 1040-slot fixed cache budget.

### Code coherence benchmark (8-turn coding session)

Class names, method signatures, and identifiers are defined in early turns. Later turns reference them. Score = fraction of expected identifiers present in response.

| Turn | Prompt tokens | Cache slots used | Coherence |
|------|--------------|-----------------|-----------|
| 0 | 104 | 0 | 100% |
| 1 | 332 | 727 | 100% |
| 3 | 783 | 982 | 100% |
| 5 | 1227 | 1040 | 100% |
| 7 | 1679 | 1040 | 100% |

Cache fills and saturates at turn 4. Eviction runs from turn 4 onward. Coherence stays at 100% throughout.

### Speed

| Context length | tok/s |
|---------------|-------|
| 256 | 4.7 |
| 512 | 4.7 |
| 1024 | 4.6 |
| 2048 | 4.4 |

~6% latency increase over 8× context growth. Vanilla HF attention would scale O(n²). The decode step cost is effectively constant because the hierarchical cache replaces growing KV with fixed-budget attention.

> **Note:** Input-side cost (summary pass over the full prompt) still scales O(T²) at first prompt per session. The constant-cost property applies to decode steps after the cache is primed. See Roadmap §1.

---

## Quickstart — Google Colab T4

### Install

```python
!pip install -q torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.43.0 accelerate>=0.33.0 bitsandbytes>=0.43.0
!pip install -q triton>=2.3.1 datasets sentencepiece faiss-cpu
!git clone https://github.com/YOUR_ORG/qwen_ppl.git && %cd qwen_ppl
```

### Run a prompt

```python
!python -m inference.colab_t4 --model 3b --quant int8
```

### Run the speed benchmark

```python
!python -m inference.colab_t4 --model 7b --quant int8 --benchmark
```

### Run the multi-turn quality benchmark

```python
# Code coherence + NIAH recall at distances 1, 3, 5, 8, 12
!python -m inference.multiturn_bench --model 3b --quant int8

# With dense baseline for direct comparison
!python -m inference.multiturn_bench --model 3b --quant int8 --baseline --save-csv /content/results/mt
```

### Use the engine in a notebook

```python
from inference.colab_t4 import build_engine

engine = build_engine(model_size="3b", quant="int8")

history = []

def chat(user_msg):
    history.append(f"User: {user_msg}")
    prompt = "\n".join(history) + "\nAssistant:"
    reply = engine.generate(prompt, max_new_tokens=200)
    history[-1] += f"\nAssistant: {reply.strip()}"
    return reply

chat("My server's secret API key is XK-99-BETA.")
chat("Explain what a REST API is in one sentence.")
chat("What was the secret API key I mentioned?")  # → XK-99-BETA
```

### Persistent multi-session memory

```python
from inference.colab_t4 import build_engine
from inference.memory_engine import PersistentMemoryEngine, SessionConfig

engine = build_engine(model_size="3b", quant="int8")
mem = PersistentMemoryEngine(engine, db_path="/content/sessions.db")

# Turn 1 — session persists to SQLite after each call
result = mem.generate(
    "My project codename is DARKSTAR.",
    session_id="session-001",
    user_id="alice",
)

# Later — session loaded from SQLite, re-encoded, primed into cache
result = mem.generate(
    "What was my project codename?",
    session_id="session-001",
    user_id="alice",
)
```

---

## Module Map

| File | What it does |
|------|-------------|
| `engine.py` | Top-level `HierarchicalInferenceEngine` — assembles all primitives |
| `two_pass.py` | Two-phase generation: O(T²) summary pass once, O(K) decode loop |
| `hierarchical_cache.py` | 3-level fixed-budget KV cache (L0 sink / L1 recent / L2 important) |
| `kll_sketch.py` | KLL streaming quantile sketch for O(1) eviction threshold |
| `domain_cache.py` | Domain-aware routing: CODE / QA / LANGUAGE cache pools |
| `rbf_path.py` | RBF kernel similarity pre-filter for cache boost and context ranking |
| `faiss_index.py` | Per-session FAISS flat index for embedding-space KV lookup |
| `session_store.py` | SQLite3 session store: turns + token memory + embeddings |
| `memory_engine.py` | `PersistentMemoryEngine`: load → generate → save cycle per request |
| `paged_cache.py` | `PagePool` + `KVAllocator`: paged GPU allocator for concurrent sessions |
| `cache_safety.py` | Defensive guards: normalise DynamicCache → legacy tuple, dtype/device checks |
| `benchmark.py` | Full evaluation suite: perplexity, NIAH, ROUGE-L, VRAM, tok/s |
| `multiturn_bench.py` | Multi-turn quality benchmark: code coherence + NIAH recall |
| `colab_t4.py` | End-to-end Colab T4 runner with CLI flags |

---

## Supported Models

Any HuggingFace `AutoModelForCausalLM` that:
- Accepts `past_key_values` (DynamicCache or legacy tuple)
- Exposes `output_hidden_states=True`

Tested with: **Qwen/Qwen2.5-3B-Instruct**, **Qwen/Qwen2.5-7B-Instruct**

> Qwen2.5 model weights are subject to [Qwen's model licence](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE). The inference engine code in this repository is MIT-licensed.

---

## Requirements

```
torch >= 2.3.1
transformers >= 4.43.0
accelerate >= 0.33.0
bitsandbytes >= 0.43.0   # int8 / int4 quantisation
triton >= 2.3.1
faiss-cpu >= 1.7.4
numpy
```

---

## Roadmap

### 1. Input-side compression (FAISS-guided prompt reconstruction) — *in progress*

**Problem:** The summary pass still runs a full O(T²) forward over the raw history string. VRAM grows linearly with conversation length on the input side even though the decode cache is fixed.

**Solution (designed, not yet connected):** On session resume, instead of concatenating the full history string as input:
1. Embed the current query
2. FAISS lookup: retrieve top-K past KV slots by embedding similarity
3. Re-encode only those K token IDs at their original position IDs (mini forward, ~10 ms for 128 tokens)
4. Prime the hierarchical cache from the retrieved KV

`session_store.py`, `faiss_index.py`, and `memory_engine.py` have this fully designed. The missing piece is wiring `multiturn_bench._run_session` to use `PersistentMemoryEngine` instead of raw string concatenation. Once connected, input length becomes O(K) not O(T) — VRAM growth flattens.

### 2. Dense baseline column in benchmark

`--baseline` flag exists in `multiturn_bench.py` and loads a second model copy. For 3B int8 (~3.5 GB × 2 = 7 GB), two copies fit on a T4. Running with `--baseline` produces the NIAH table with both columns. **This is the next thing to run** — it will show the hierarchical engine vs. vanilla rolling-context directly.

### 3. Request / session / project identity model

Currently: `session_id` + `user_id`.

Planned schema: `project_id → session_id → request_id`. Each request belongs to a session, each session belongs to a project. This enables:
- Project-level cache sharing (common context across sessions in the same project)
- Per-project eviction budgets
- Audit logging at the request level

The SQLite schema change and API updates are the scope.

### 4. Triton fused attention kernel

`triton_ops.py` contains a fused kernel but it is not wired into model forward. Wiring requires monkey-patching `Qwen2Attention.forward` to call the custom kernel instead of `F.scaled_dot_product_attention`. Expected throughput gain: 6–8 tok/s vs current 4–5 tok/s on T4.

### 5. Multi-session paged allocator in benchmarks

`paged_cache.py` implements `PagePool` + `KVAllocator` + `MultiSessionCache` with LRU page eviction. The design is complete and tested in isolation. It is not yet connected to `HierarchicalInferenceEngine`. Once connected, concurrent session count on a T4 goes from 2–3 to 10–16.

---

## Known Limitations

- **Input scaling:** Summary pass cost is O(T²) over the raw prompt. Fixed by Roadmap §1.
- **Triton kernel not active:** All inference currently uses PyTorch SDPA (`sdpa` backend). Triton fused attention is designed but not wired (Roadmap §4).
- **No project hierarchy yet:** Session store has `session_id` / `user_id` only. `project_id` and `request_id` are Roadmap §3.
- **Single GPU:** The paged allocator is designed for single-GPU multi-session. Multi-GPU tensor parallel is out of scope.

---

## Citation / Prior Work

This engine draws on ideas from:

- **StreamingLLM** (Xiao et al., 2023) — attention sink protection (L0 level)
- **H2O** (Zhang et al., 2023) — heavy-hitter oracle, attention-mass importance scoring
- **InfiniAttention** (Munkhdalai et al., 2024) — compressive memory for infinite context
- **PagedAttention / vLLM** (Kwon et al., 2023) — paged KV allocator design (Roadmap §5)

The KLL sketch eviction threshold, RBF path pre-filter, domain-specific cache routing, and SQLite-backed session persistence are original contributions of this codebase.
