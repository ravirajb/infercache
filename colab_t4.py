"""
inference/colab_t4.py
---------------------
End-to-end runner for HierarchicalInferenceEngine on Google Colab T4.

Targets
-------
  GPU  : Tesla T4   (16 GB VRAM, sm_75, NO bfloat16, ~320 GB/s BW)
  Model: Qwen/Qwen2.5-3B-Instruct   (~6 GB fp16, ~3.5 GB int8, ~2 GB int4)
         Qwen/Qwen2.5-7B-Instruct   (~14 GB fp16, ~7 GB int8, ~4 GB int4)

Quick-start: Open a Colab notebook, paste each section in a separate cell,
or run the file directly:
    !python -m inference.colab_t4 --model 3b
    !python -m inference.colab_t4 --model 7b --quant int8 --benchmark

Usage flags:
  --model {3b,7b}          which Qwen variant (default: 3b)
  --quant {fp16,int8,int4} quantisation level (default: int8)
  --max-tokens N           tokens to generate (default: 512)
  --prompt FILE            read prompt from a .txt or .py file
  --benchmark              run speed benchmark instead of code examples
  --no-code-mode           disable code-optimised cache params
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# ═  CELL 1 — Install commands (print and copy into a shell cell)           ═
# ════════════════════════════════════════════════════════════════════════════

INSTALL_COMMANDS = """\
!pip install -q --upgrade pip
!pip install -q torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.43.0 accelerate>=0.33.0
!pip install -q bitsandbytes>=0.43.0
!pip install -q triton>=2.3.1
!pip install -q datasets sentencepiece faiss-cpu
"""

# ════════════════════════════════════════════════════════════════════════════
# ═  CELL 2 — Upload / Clone the repo                                       ═
# ════════════════════════════════════════════════════════════════════════════

CLONE_COMMANDS = """\
# Option A: clone from GitHub
# !git clone https://github.com/YOUR_ORG/qwen_ppl.git && %cd qwen_ppl

# Option B: upload a zip
# from google.colab import files; files.upload()
# !unzip -q qwen_ppl.zip && %cd qwen_ppl

# Option C: mount Google Drive
# from google.colab import drive; drive.mount('/content/drive')
# %cd /content/drive/MyDrive/qwen_ppl
"""

# ════════════════════════════════════════════════════════════════════════════
# ═  GPU health check                                                        ═
# ════════════════════════════════════════════════════════════════════════════

def check_gpu() -> None:
    import sys
    import torch

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU detected. Change Runtime → T4 GPU and restart.")
        sys.exit(1)

    name  = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {name}")
    print(f"VRAM: {total:.1f} GB")
    print(f"CUDA: {torch.version.cuda}   PyTorch: {torch.__version__}")

    if "T4" in name:
        print("T4 detected → using float16 (T4 does NOT support bfloat16)")
    elif any(g in name for g in ("A100", "V100", "L4", "A10")):
        print(f"{name} detected → bfloat16 supported; edit cfg.dtype if desired")

    try:
        import triton
        print(f"Triton: {triton.__version__}  OK")
    except ImportError:
        print("Triton: not installed — fused kernels fall back to PyTorch SDPA")


# ════════════════════════════════════════════════════════════════════════════
# ═  Build engine                                                            ═
# ════════════════════════════════════════════════════════════════════════════

_MODEL_IDS = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

# Estimated peak VRAM (model weights + code cache × 3 domains)
_VRAM_GUIDE = {
    ("3b", "fp16"): 6.5,
    ("3b", "int8"): 3.8,
    ("3b", "int4"): 2.2,
    ("7b", "fp16"): 14.5,
    ("7b", "int8"): 7.8,
    ("7b", "int4"): 4.5,
}


def build_engine(
    model_size: str  = "3b",
    quant:      str  = "int8",
    device:     str  = "cuda",
    code_mode:  bool = True,
    max_new_tokens: int = 512,
):
    """
    Build and return a HierarchicalInferenceEngine.

    Parameters
    ----------
    model_size    : "3b" | "7b"
    quant         : "fp16" | "int8" | "int4"
    device        : "cuda" | "cpu"
    code_mode     : apply code-optimised cache settings (recommended)
    max_new_tokens: maximum tokens to generate per call
    """
    import torch
    from inference import HierarchicalInferenceEngine, build_qwen_config
    from inference.code_cache import code_engine_config

    if model_size not in _MODEL_IDS:
        raise ValueError(f"model_size must be '3b' or '7b', got {model_size!r}")

    model_name = _MODEL_IDS[model_size]
    est_vram   = _VRAM_GUIDE.get((model_size, quant), 0.0)
    print(f"Loading {model_name}  [{quant}]  est. VRAM ≈ {est_vram:.1f} GB")

    # T4 does NOT support bfloat16 — always use float16
    cfg       = build_qwen_config(model_name, device=device)
    cfg.dtype = torch.float16
    cfg.max_new_tokens = max_new_tokens

    if code_mode:
        cfg = code_engine_config(cfg, max_new_tokens=max_new_tokens)
        print(f"Code-mode ON: sink=16, recent=512, important=512, "
              f"decay=0.999, max_new_tokens={max_new_tokens}")
    else:
        cfg.max_new_tokens = max_new_tokens
        print(f"Default cache: sink={cfg.sink_size}, recent={cfg.recent_size}, "
              f"important={cfg.important_size}, max_new_tokens={max_new_tokens}")

    engine = HierarchicalInferenceEngine.from_pretrained(
        model_name,
        cfg,
        load_in_8bit = (quant == "int8"),
        load_in_4bit = (quant == "int4"),
    )

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved  = torch.cuda.memory_reserved(0)  / 1e9
    print(f"Ready. VRAM allocated: {allocated:.2f} GB  reserved: {reserved:.2f} GB")
    return engine


# ════════════════════════════════════════════════════════════════════════════
# ═  Code generation examples                                                ═
# ════════════════════════════════════════════════════════════════════════════

CODE_PROMPTS = [
    # 1. Pure generation
    "Write a Python function that implements merge sort.\nInclude type hints and a docstring.\n",

    # 2. Code completion with long context (tests cache reuse)
    """\
Here is a partially implemented binary search tree in Python:

class TreeNode:
    def __init__(self, val: int):
        self.val = val; self.left = None; self.right = None

class BST:
    def __init__(self): self.root = None

    def insert(self, val: int) -> None:
        if self.root is None:
            self.root = TreeNode(val); return
        cur = self.root
        while True:
            if val < cur.val:
                if cur.left is None:
                    cur.left = TreeNode(val); return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = TreeNode(val); return
                cur = cur.right

    def search(self, val: int) -> bool:
        cur = self.root
        while cur:
            if val == cur.val: return True
            cur = cur.left if val < cur.val else cur.right
        return False

Now implement:
  - delete(val)  — remove node, preserve BST property
  - inorder()    — return sorted list of all values
  - height()     — return height of the tree
""",

    # 3. Triton kernel explanation
    """\
Explain what this Triton kernel does and how it differs from vanilla PyTorch attention:

@triton.jit
def _chunk_attn_fwd(Q, K, V, Out, stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd, stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0); off_h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + offs_m[:, None] * stride_qm + off_h * stride_qh + offs_d[None, :])
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(K + (start_n + offs_n)[None, :] * stride_km + off_h * stride_kh + offs_d[:, None])
        s = tl.dot(q, k)
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(tl.exp(s - m_new[:, None]), 1); m_i = m_new
        v = tl.load(V + (start_n + offs_n)[None, :] * stride_vm + off_h * stride_vh + offs_d[:, None])
        acc += tl.dot(tl.exp(s - m_i[:, None]), v)
    tl.store(Out + offs_m[:, None] * stride_om + off_h * stride_oh + offs_d[None, :], acc / l_i[:, None])
""",
]


def run_code_examples(engine, prompts=None, verbose: bool = True) -> None:
    """Run each code prompt through the engine and print results."""
    import time
    prompts = prompts or CODE_PROMPTS

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"PROMPT {i}/{len(prompts)}:")
        print(prompt[:200] + ("…" if len(prompt) > 200 else ""))
        print("-" * 70)

        t0      = time.perf_counter()
        output  = engine.generate(prompt)
        elapsed = time.perf_counter() - t0

        print(output[:500] + ("…" if not verbose and len(output) > 500 else "") if not verbose else output)

        import dataclasses
        raw   = engine.cache_stats().get("CODE")
        cstat = dataclasses.asdict(raw) if raw is not None else {}
        cstat.pop("total_tokens_seen", None)
        cstat.pop("total_evictions",   None)
        print(f"\n[{elapsed:.1f}s]  CODE cache: {cstat}")


# ════════════════════════════════════════════════════════════════════════════
# ═  Quick speed + memory benchmark                                          ═
# ════════════════════════════════════════════════════════════════════════════

def quick_benchmark(engine, context_lengths=None) -> None:
    """Measure tokens/sec and VRAM at increasing context lengths."""
    import gc, time
    import torch

    context_lengths = context_lengths or [256, 512, 1024, 2048]
    base_prompt     = CODE_PROMPTS[0]

    print("\n=== Quick Speed Benchmark ===")
    print(f"{'Context':>10}  {'Time (s)':>10}  {'tok/s':>8}  {'VRAM (GB)':>12}")
    print("-" * 47)

    for ctx in context_lengths:
        filler = "x = 1\n" * max(1, (ctx - 60) // 6)
        prompt = filler + "\n" + base_prompt

        engine.reset()
        gc.collect()
        torch.cuda.empty_cache()

        t0     = time.perf_counter()
        output = engine.generate(prompt)
        elapsed = time.perf_counter() - t0

        vram    = torch.cuda.memory_allocated(0) / 1e9
        n_words = len(output.split())
        tps     = n_words / max(elapsed, 1e-6)

        print(f"{ctx:>10}  {elapsed:>10.2f}  {tps:>8.1f}  {vram:>10.2f} GB")


# ════════════════════════════════════════════════════════════════════════════
# ═  Triton warmup                                                           ═
# ════════════════════════════════════════════════════════════════════════════

def triton_warmup(engine) -> None:
    """
    Run a short dummy prompt to trigger Triton JIT compilation before timing.
    On T4 (sm_75), first-call compilation can take 10–30 s.
    """
    import time
    print("Warming up Triton kernels (may take 10-30 s on first run)…")
    t0 = time.perf_counter()
    engine.generate("def hello(): pass\n" * 8)
    print(f"  Warmup done in {time.perf_counter() - t0:.1f} s")
    engine.reset()


# ════════════════════════════════════════════════════════════════════════════
# ═  Interactive streaming (for notebook cells)                              ═
# ════════════════════════════════════════════════════════════════════════════

def interactive_stream(engine, prompt: str) -> str:
    """
    Stream tokens to stdout token-by-token.
    Call from a Colab notebook cell::

        from inference.colab_t4 import interactive_stream
        interactive_stream(engine, "Write a Python quicksort.")
    """
    import time
    result: list[str] = []
    t0 = time.perf_counter()

    for token in engine.stream_generate(prompt):
        result.append(token)
        print(token, end="", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n\n[Done — {len(result)} tokens in {elapsed:.1f} s  "
          f"({len(result)/max(elapsed,1e-6):.1f} tok/s)]")
    return "".join(result)


# ════════════════════════════════════════════════════════════════════════════
# ═  CLI entry point                                                         ═
# ════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="HierarchicalInferenceEngine — T4 Colab runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        choices=["3b", "7b"], default="3b")
    p.add_argument("--quant",        choices=["fp16", "int8", "int4"], default="int8")
    p.add_argument("--max-tokens",   type=int, default=512, dest="max_tokens")
    p.add_argument("--prompt",       type=str, default=None,
                   help="Path to a .txt or .py file to use as the prompt")
    p.add_argument("--benchmark",    action="store_true",
                   help="Run speed benchmark instead of demo prompts")
    p.add_argument("--no-code-mode", action="store_true",
                   help="Disable code-optimised cache settings")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    check_gpu()

    engine = build_engine(
        model_size     = args.model,
        quant          = args.quant,
        code_mode      = not args.no_code_mode,
        max_new_tokens = args.max_tokens,
    )

    triton_warmup(engine)

    if args.benchmark:
        quick_benchmark(engine)
    elif args.prompt:
        with open(args.prompt) as f:
            prompt = f.read()
        print(engine.generate(prompt))
    else:
        run_code_examples(engine)


if __name__ == "__main__":
    main()
