"""
inference/multiturn_bench.py
-----------------------------
Multi-turn conversation benchmark for HierarchicalInferenceEngine.

Why multi-turn is the key use-case
-----------------------------------
Standard causal-LM KV cache grows O(T) per turn, so after N turns the
context is T_1 + T_2 + ... + T_N tokens long.  At turn 20, a 200-token
turn means 4000 tokens of KV state — 4× the VRAM and 4× the per-step
attention cost compared to turn 1.  The model eventually hits its context
window limit and the session must be truncated, silently losing earlier
context.

HierarchicalInferenceEngine maintains a FIXED-BUDGET cache (O(budget))
regardless of how many turns have elapsed.  This benchmark measures:

Quality axes
------------
  1. Context retention (multi-turn NIAH)
     Plant a fact in turn K, ask about it in turn K+D.
     Measure recall@D: fraction retrieved vs turns-ago-planted distance D.
     Baseline: rolling-context model (keeps last W tokens).

  2. Code coherence score
     Define a class / variables in early turns.
     Later turns add methods or use those names.
     Check if the response uses the correct identifier names rather than
     hallucinating new ones.  Score = exact-match fraction over expected
     identifiers.

  3. Cross-turn perplexity (continuation PPL)
     Given N prior turns as context, measure perplexity of turn N+1.
     Lower = model has better access to prior context.

Memory axes
-----------
  4. VRAM profile over turns
     Record peak VRAM after each turn for both baseline and hierarchical.
     Expected: baseline grows linearly, hierarchical stays flat.

  5. KV cache fill breakdown
     Report sink / recent / important fill counts after each turn.
     Demonstrates that the cache is being utilised, not just empty.

Speed axes
----------
  6. Tokens/sec over turns
     Baseline slows down because attention cost grows with context length.
     Hierarchical should remain approximately constant.

  7. Per-turn first-token latency (TTFT)
     TTFT should stay approximately constant for hierarchical.

Running
-------
  # Minimal (no dense baseline, no GPU):
  from inference.multiturn_bench import MultiTurnBench, print_report
  bench = MultiTurnBench(engine)
  results = bench.run()
  print_report(results)

  # Full comparison with dense baseline:
  bench = MultiTurnBench(engine, dense_model=model)
  results = bench.run()
  bench.save_csv("results/multiturn.csv")

  # From CLI on Colab:
  !python -m inference.multiturn_bench --model 3b --quant int8 --turns 30
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "MultiTurnBench",
    "TurnResult",
    "MultiTurnResult",
    "print_report",
]


# ══════════════════════════════════════════════════════════════════════════════
# ── Conversation scripts ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Script 1 — Code session
# A realistic multi-turn coding session where the model must remember
# class names, method signatures, and variable names across turns.
# ---------------------------------------------------------------------------
CODE_SESSION: List[Dict] = [
    {   # turn 0: define the class
        "role": "user",
        "content": (
            "Let's build a class called `TokenBudgetManager` in Python. "
            "It should have:\n"
            "  - `__init__(self, budget: int)` storing `self.budget` and `self.used = 0`\n"
            "  - `allocate(self, n: int) -> bool` — returns True if enough budget remains, "
            "else False, increments `self.used` on success\n"
            "  - `remaining(self) -> int` — returns budget minus used\n"
            "Just write the class, no extra explanation."
        ),
        "expected_identifiers": ["TokenBudgetManager", "budget", "used", "allocate", "remaining"],
    },
    {   # turn 1: add method
        "role": "user",
        "content": (
            "Add a `reset(self)` method to `TokenBudgetManager` that sets `self.used` back to 0."
        ),
        "expected_identifiers": ["TokenBudgetManager", "reset", "used"],
    },
    {   # turn 2: unrelated filler turn (tests that filler doesn't evict core context)
        "role": "user",
        "content": "Write a one-liner Python function that reverses a string.",
        "expected_identifiers": [],
    },
    {   # turn 3: reference early class again
        "role": "user",
        "content": (
            "Now write a subclass of `TokenBudgetManager` called `LoggingBudgetManager` "
            "that overrides `allocate` to print a message each time allocation succeeds."
        ),
        "expected_identifiers": ["LoggingBudgetManager", "TokenBudgetManager", "allocate"],
    },
    {   # turn 4: filler
        "role": "user",
        "content": "Explain what a Python decorator is.",
        "expected_identifiers": [],
    },
    {   # turn 5: reference from even further back
        "role": "user",
        "content": (
            "Write a unit test for `TokenBudgetManager.allocate` using `unittest.TestCase`. "
            "Test that `allocate` returns False when the budget is exhausted."
        ),
        "expected_identifiers": ["TokenBudgetManager", "allocate", "budget"],
    },
    {   # turn 6: filler
        "role": "user",
        "content": "What is the difference between `__str__` and `__repr__` in Python?",
        "expected_identifiers": [],
    },
    {   # turn 7: final reference (deepest recall test — 7 turns since definition)
        "role": "user",
        "content": (
            "Write a context manager `BudgetContext` that wraps a `TokenBudgetManager` "
            "and automatically calls `reset()` on exit."
        ),
        "expected_identifiers": ["BudgetContext", "TokenBudgetManager", "reset"],
    },
]

# ---------------------------------------------------------------------------
# Script 2 — Multi-turn NIAH (needle in a haystack across turns)
# Plant a secret fact in turn K, ask about it D turns later.
# ---------------------------------------------------------------------------
_FACT_TEMPLATE = (
    "Remember this important fact: the secret project code is {code}. "
    "Keep this in mind for later."
)
_RECALL_TEMPLATE = "What was the secret project code I mentioned earlier?"

# Distances at which we test recall (turns after planting)
_NIAH_DISTANCES = [1, 3, 5, 8, 12]
_NIAH_CODE = "FALCON-7749"       # the needle

def _build_niah_session(distance: int) -> List[Dict]:
    """Build a session that plants the needle at turn 0 and asks at turn `distance`."""
    fillers = [
        "Write a bubble sort implementation.",
        "Explain what machine learning is.",
        "What is the difference between a list and a tuple in Python?",
        "Write a function that checks if a number is prime.",
        "How does garbage collection work in Python?",
        "Write a function to compute Fibonacci numbers iteratively.",
        "What is the difference between deep copy and shallow copy?",
        "Write a regex to extract all email addresses from a string.",
        "Explain what a hash table is.",
        "Write a decorator that measures function execution time.",
        "What is the CAP theorem in distributed systems?",
        "Write a function that flattens a nested list.",
    ]
    session = []
    session.append({
        "role": "user",
        "content": _FACT_TEMPLATE.format(code=_NIAH_CODE),
        "is_plant": True,
    })
    for i in range(distance - 1):
        session.append({
            "role": "user",
            "content": fillers[i % len(fillers)],
        })
    session.append({
        "role": "user",
        "content": _RECALL_TEMPLATE,
        "is_recall": True,
        "expected_code": _NIAH_CODE,
    })
    return session


# ══════════════════════════════════════════════════════════════════════════════
# ── Result containers ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurnResult:
    turn_idx:          int
    prompt_tokens:     int
    output_tokens:     int
    elapsed_ms:        float
    ttft_ms:           float          # first-token time (proxy: elapsed / output_tokens)
    tps:               float          # tokens / second
    vram_gb:           float          # peak VRAM after this turn
    cache_fill:        int            # total occupied slots
    sink_fill:         int
    recent_fill:       int
    important_fill:    int
    coherence_score:   float          # identifier match (0-1); -1 if N/A
    response_preview:  str            # first 120 chars


@dataclass
class NIAHTurnResult:
    distance:    int
    retrieved:   bool
    hier_answer: str
    dense_answer: Optional[str] = None

    @property
    def hier_score(self) -> float:
        return 1.0 if self.retrieved else 0.0


@dataclass
class MultiTurnResult:
    engine_label:     str
    turns:            List[TurnResult]           = field(default_factory=list)
    niah_results:     List[NIAHTurnResult]        = field(default_factory=list)
    dense_turns:      List[TurnResult]            = field(default_factory=list)  # baseline
    cache_budget:     int                         = 0
    n_turns:          int                         = 0


# ══════════════════════════════════════════════════════════════════════════════
# ── Scoring helpers ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _identifier_score(response: str, expected: List[str]) -> float:
    """Fraction of expected identifier names that appear in the response."""
    if not expected:
        return -1.0          # N/A for this turn
    hits = sum(1 for s in expected if s in response)
    return hits / len(expected)


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _vram_gb(device: str = "cuda") -> float:
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def _reset_vram(device: str = "cuda") -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ══════════════════════════════════════════════════════════════════════════════
# ── Dense baseline helper ─────────────────────────════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class _RollingContextBaseline:
    """
    Naive rolling-window baseline: concatenates all prior turns into a
    single context up to `max_context_tokens` tokens, truncating oldest
    turns first.  This simulates how a standard LLM chat wrapper works.
    """

    def __init__(self, model, tokenizer, max_context_tokens: int = 3000,
                 max_new_tokens: int = 256, device: str = "cuda"):
        self.model              = model
        self.tokenizer          = tokenizer
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens     = max_new_tokens
        self.device             = torch.device(device)
        self._history: List[str] = []   # list of "User: ...\nAssistant: ..." strings

    def generate(self, prompt: str) -> str:
        self._history.append(f"User: {prompt}")
        context = self._build_context()
        ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            out = self.model.generate(
                ids,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,    # greedy for reproducibility
                pad_token_id   = self.tokenizer.eos_token_id,
            )
        new_ids  = out[0][ids.shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self._history[-1] += f"\nAssistant: {response}"
        return response

    def reset(self) -> None:
        self._history.clear()

    def _build_context(self) -> str:
        """Build prompt, trimming oldest turns if over budget."""
        # Walk from oldest, accumulate until we'd exceed budget
        kept, budget = [], self.max_context_tokens
        for turn in reversed(self._history):
            n = _token_count(self.tokenizer, turn)
            if n > budget:
                break
            kept.insert(0, turn)
            budget -= n
        return "\n".join(kept) + "\nAssistant:"


# ══════════════════════════════════════════════════════════════════════════════
# ── Main benchmark class ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class MultiTurnBench:
    """
    Multi-turn quality + memory + speed benchmark.

    Parameters
    ----------
    engine       : HierarchicalInferenceEngine
    dense_model  : optional raw HuggingFace model for baseline comparison
    device       : "cuda" | "cpu"
    max_new_tokens: tokens to generate per turn
    """

    def __init__(
        self,
        engine,
        dense_model   = None,
        tokenizer     = None,       # defaults to engine.tokenizer
        device        : str  = "cuda",
        max_new_tokens: int  = 200,
    ):
        self.engine         = engine
        self.dense_model    = dense_model
        self.tokenizer      = tokenizer or engine.tokenizer
        self.device         = device
        self.max_new_tokens = max_new_tokens

        self._baseline: Optional[_RollingContextBaseline] = None
        if dense_model is not None:
            self._baseline = _RollingContextBaseline(
                dense_model, self.tokenizer,
                max_context_tokens = 3000,
                max_new_tokens     = max_new_tokens,
                device             = device,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        n_code_sessions:  int  = 1,
        niah_distances:   List[int] = None,
        verbose:          bool = True,
    ) -> MultiTurnResult:
        """
        Run the full multi-turn benchmark suite.

        Returns a MultiTurnResult with per-turn stats for both
        hierarchical engine and (if available) the dense baseline.
        """
        from inference.engine import HierarchicalInferenceEngine
        label = type(self.engine).__name__
        result = MultiTurnResult(
            engine_label = label,
            cache_budget = getattr(
                getattr(self.engine, "cfg", None), "total_budget", 0
            ) or sum([
                getattr(getattr(self.engine, "cfg", None), k, 0)
                for k in ("sink_size", "recent_size", "important_size")
            ]),
        )
        result.cache_budget = result.cache_budget or 1040  # code-mode default

        # ── 1. Code session ───────────────────────────────────────────────
        if verbose:
            print("\n" + "═"*70)
            print(" CODE SESSION BENCHMARK")
            print("═"*70)

        self.engine.reset()
        if self._baseline:
            self._baseline.reset()

        hier_turns  = self._run_session(CODE_SESSION, verbose=verbose, label="HIER")
        dense_turns: List[TurnResult] = []
        if self._baseline:
            if verbose:
                print("\n--- Baseline (rolling context) ---")
            self._baseline.reset()
            dense_turns = self._run_session_baseline(CODE_SESSION, verbose=verbose)

        result.turns       = hier_turns
        result.dense_turns = dense_turns
        result.n_turns     = len(hier_turns)

        # ── 2. Multi-turn NIAH ────────────────────────────────────────────
        distances = niah_distances if niah_distances is not None else _NIAH_DISTANCES
        if verbose:
            print("\n" + "═"*70)
            print(" MULTI-TURN NIAH BENCHMARK")
            print("═"*70)

        for dist in distances:
            nr = self._run_niah(dist, verbose=verbose)
            result.niah_results.append(nr)

        return result

    # ── Session runner (hierarchical) ─────────────────────────────────────────

    def _run_session(
        self,
        session: List[Dict],
        verbose: bool,
        label:   str = "HIER",
    ) -> List[TurnResult]:
        results: List[TurnResult] = []
        # Conversation history: list of "User: ...\nAssistant: ..." blocks.
        # Prepended to each new turn so the engine sees full context.
        history: List[str] = []

        for i, turn in enumerate(session):
            user_msg = turn["content"]
            expected = turn.get("expected_identifiers", [])

            # Build the full prompt: prior history + current user message
            history.append(f"User: {user_msg}")
            prompt = "\n".join(history) + "\nAssistant:"

            _reset_vram(self.device)
            t0 = time.perf_counter()
            response = self.engine.generate(prompt, max_new_tokens=self.max_new_tokens)
            elapsed  = (time.perf_counter() - t0) * 1000

            # Append model response to history for the next turn
            history[-1] += f"\nAssistant: {response.strip()}"

            n_out  = _token_count(self.tokenizer, response)
            tps    = (n_out / max(elapsed / 1000, 1e-6))
            ttft   = elapsed / max(n_out, 1)     # proxy (not true streaming TTFT)
            vram   = _vram_gb(self.device)
            score  = _identifier_score(response, expected)

            # Cache fill from engine
            raw_stats   = self.engine.cache_stats()
            active_stat = (
                raw_stats.get("CODE") or
                raw_stats.get("LANGUAGE") or
                next(iter(raw_stats.values()), None)
            )

            if active_stat is not None and hasattr(active_stat, "current_fill"):
                cs = active_stat
            else:
                cs = type("_CS", (), {
                    "current_fill": 0, "sink_fill": 0,
                    "recent_fill": 0, "important_fill": 0,
                })()

            tr = TurnResult(
                turn_idx        = i,
                prompt_tokens   = _token_count(self.tokenizer, prompt),
                output_tokens   = n_out,
                elapsed_ms      = elapsed,
                ttft_ms         = ttft,
                tps             = tps,
                vram_gb         = vram,
                cache_fill      = cs.current_fill,
                sink_fill       = cs.sink_fill,
                recent_fill     = cs.recent_fill,
                important_fill  = cs.important_fill,
                coherence_score = score,
                response_preview= response[:120].replace("\n", " "),
            )
            results.append(tr)

            if verbose:
                score_str = f"{score:.0%}" if score >= 0 else " N/A"
                print(
                    f"  [{label}] turn={i:2d}  "
                    f"toks={n_out:4d}  "
                    f"t={elapsed/1000:5.1f}s  "
                    f"tps={tps:5.1f}  "
                    f"vram={vram:.2f}GB  "
                    f"cache={cs.current_fill:4d}  "
                    f"coherence={score_str}"
                )

        return results

    # ── Session runner (dense baseline) ───────────────────────────────────────

    def _run_session_baseline(
        self,
        session: List[Dict],
        verbose: bool,
    ) -> List[TurnResult]:
        if self._baseline is None:
            return []
        results: List[TurnResult] = []
        history: List[str] = []

        for i, turn in enumerate(session):
            user_msg = turn["content"]
            expected = turn.get("expected_identifiers", [])

            history.append(f"User: {user_msg}")
            prompt = "\n".join(history) + "\nAssistant:"

            _reset_vram(self.device)
            t0 = time.perf_counter()
            response = self._baseline.generate(prompt)
            elapsed  = (time.perf_counter() - t0) * 1000

            history[-1] += f"\nAssistant: {response.strip()}"

            n_out = _token_count(self.tokenizer, response)
            tps   = (n_out / max(elapsed / 1000, 1e-6))
            vram  = _vram_gb(self.device)
            score = _identifier_score(response, expected)

            tr = TurnResult(
                turn_idx        = i,
                prompt_tokens   = _token_count(self.tokenizer, prompt),
                output_tokens   = n_out,
                elapsed_ms      = elapsed,
                ttft_ms         = elapsed / max(n_out, 1),
                tps             = tps,
                vram_gb         = vram,
                cache_fill      = 0,    # N/A for rolling baseline
                sink_fill       = 0,
                recent_fill     = 0,
                important_fill  = 0,
                coherence_score = score,
                response_preview= response[:120].replace("\n", " "),
            )
            results.append(tr)

            if verbose:
                score_str = f"{score:.0%}" if score >= 0 else " N/A"
                print(
                    f"  [BASE] turn={i:2d}  "
                    f"toks={n_out:4d}  "
                    f"t={elapsed/1000:5.1f}s  "
                    f"tps={tps:5.1f}  "
                    f"vram={vram:.2f}GB  "
                    f"coherence={score_str}"
                )
        return results

    # ── Multi-turn NIAH ────────────────────────────────────────────────────────

    def _run_niah(self, distance: int, verbose: bool) -> NIAHTurnResult:
        session  = _build_niah_session(distance)
        self.engine.reset()

        hier_answer = ""
        hier_history: List[str] = []
        for turn in session:
            hier_history.append(f"User: {turn['content']}")
            prompt = "\n".join(hier_history) + "\nAssistant:"
            r = self.engine.generate(prompt, max_new_tokens=80)
            hier_history[-1] += f"\nAssistant: {r.strip()}"
            if turn.get("is_recall"):
                hier_answer = r

        retrieved = _NIAH_CODE in hier_answer

        # Optionally run baseline too
        dense_answer: Optional[str] = None
        if self._baseline:
            self._baseline.reset()
            dense_history: List[str] = []
            for turn in session:
                dense_history.append(f"User: {turn['content']}")
                prompt = "\n".join(dense_history) + "\nAssistant:"
                r = self._baseline.generate(prompt)
                dense_history[-1] += f"\nAssistant: {r.strip()}"
                if turn.get("is_recall"):
                    dense_answer = r

        nr = NIAHTurnResult(
            distance     = distance,
            retrieved     = retrieved,
            hier_answer  = hier_answer[:120],
            dense_answer = dense_answer[:120] if dense_answer else None,
        )

        if verbose:
            tag     = "✓" if retrieved else "✗"
            d_tag   = ""
            if dense_answer is not None:
                d_retrieved = _NIAH_CODE in dense_answer
                d_tag = f"  baseline={'✓' if d_retrieved else '✗'}"
            print(f"  NIAH distance={distance:2d}  hier={tag}{d_tag}  "
                  f"answer_preview='{hier_answer[:60]}'")

        return nr


# ══════════════════════════════════════════════════════════════════════════════
# ── Reporting ─────────────────────────────────────────────════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def print_report(result: MultiTurnResult) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "═"*70)
    print(f" MULTI-TURN BENCHMARK REPORT — {result.engine_label}")
    print(f" Cache budget: {result.cache_budget} slots")
    print("═"*70)

    # ── Per-turn table ────────────────────────────────────────────────────
    print(f"\n{'Turn':>5}  {'Prompt':>7}  {'Output':>7}  {'Time(s)':>8}  "
          f"{'tok/s':>7}  {'VRAM(GB)':>9}  {'Cache':>6}  {'Coherence':>10}")
    print("-" * 72)

    for tr in result.turns:
        score_str = f"{tr.coherence_score:.0%}" if tr.coherence_score >= 0 else "  N/A"
        print(
            f"{tr.turn_idx:>5}  "
            f"{tr.prompt_tokens:>7}  "
            f"{tr.output_tokens:>7}  "
            f"{tr.elapsed_ms/1000:>8.2f}  "
            f"{tr.tps:>7.1f}  "
            f"{tr.vram_gb:>9.2f}  "
            f"{tr.cache_fill:>6}  "
            f"{score_str:>10}"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────
    if result.turns:
        turns       = result.turns
        avg_tps     = sum(t.tps for t in turns) / len(turns)
        last_vram   = turns[-1].vram_gb
        first_vram  = turns[0].vram_gb
        vram_growth = last_vram - first_vram
        scored      = [t for t in turns if t.coherence_score >= 0]
        avg_coh     = sum(t.coherence_score for t in scored) / len(scored) if scored else 0.0

        print()
        print(f"  Avg tokens/sec      : {avg_tps:.1f}")
        print(f"  VRAM turn 0 → last  : {first_vram:.2f} → {last_vram:.2f} GB  "
              f"(Δ {vram_growth:+.2f} GB)")
        print(f"  Avg coherence score : {avg_coh:.0%}")
        print(f"  Final cache fill    : {turns[-1].cache_fill} / {result.cache_budget}")

    # ── Comparison with dense baseline ────────────────────────────────────
    if result.dense_turns:
        d = result.dense_turns
        h = result.turns
        print("\n  --- vs Dense Baseline (rolling context) ---")
        for i in range(min(len(h), len(d))):
            hi, di = h[i], d[i]
            tps_delta  = hi.tps - di.tps
            vram_delta = hi.vram_gb - di.vram_gb
            coh_delta  = (hi.coherence_score - di.coherence_score
                          if hi.coherence_score >= 0 and di.coherence_score >= 0 else None)
            coh_str    = (f"{coh_delta:+.0%}" if coh_delta is not None else " N/A")
            print(
                f"    turn {i}: tok/s {tps_delta:+.1f}  "
                f"vram {vram_delta:+.2f}GB  "
                f"coherence {coh_str}"
            )

    # ── NIAH table ────────────────────────────────────────────────────────
    if result.niah_results:
        print("\n  --- Multi-Turn NIAH (fact recall vs distance) ---")
        print(f"  {'Distance':>9}  {'Hier':>6}  {'Dense':>7}")
        print("  " + "-"*27)
        for nr in result.niah_results:
            h_tag = "✓" if nr.retrieved else "✗"
            d_tag = ("✓" if (_NIAH_CODE in (nr.dense_answer or "")) else "✗") \
                    if nr.dense_answer is not None else " N/A"
            print(f"  {nr.distance:>9}  {h_tag:>6}  {d_tag:>7}")
        hier_recall = sum(1 for n in result.niah_results if n.retrieved) / len(result.niah_results)
        print(f"\n  Hierarchical NIAH recall: {hier_recall:.0%}"
              f"  ({sum(1 for n in result.niah_results if n.retrieved)}"
              f"/{len(result.niah_results)} retrieved)")

    print()


def save_csv(result: MultiTurnResult, path: str = "results/multiturn.csv") -> None:
    """Write per-turn stats and NIAH results to CSV files."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Per-turn CSV
    turn_path = path.replace(".csv", "_turns.csv")
    with open(turn_path, "w", newline="") as f:
        fields = [f.name for f in dataclasses.fields(TurnResult)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in result.turns:
            w.writerow(dataclasses.asdict(tr))
    print(f"  Saved turn data → {turn_path}")

    # NIAH CSV
    niah_path = path.replace(".csv", "_niah.csv")
    with open(niah_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["distance", "retrieved", "hier_score", "hier_answer"])
        w.writeheader()
        for nr in result.niah_results:
            w.writerow({
                "distance" : nr.distance,
                "retrieved": nr.retrieved,
                "hier_score": nr.hier_score,
                "hier_answer": nr.hier_answer,
            })
    print(f"  Saved NIAH data  → {niah_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI entry point ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Multi-turn benchmark for HierarchicalInferenceEngine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      choices=["3b", "7b"], default="3b")
    p.add_argument("--quant",      choices=["fp16", "int8", "int4"], default="int8")
    p.add_argument("--turns",      type=int, default=8,
                   help="Number of code session turns to run")
    p.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    p.add_argument("--save-csv",   type=str, default=None, dest="save_csv",
                   help="Path prefix for output CSV files")
    p.add_argument("--no-niah",    action="store_true",
                   help="Skip multi-turn NIAH benchmark")
    p.add_argument("--baseline",   action="store_true",
                   help="Load a second copy of the model as rolling-context baseline")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from inference.colab_t4 import build_engine, check_gpu
    check_gpu()

    engine = build_engine(
        model_size     = args.model,
        quant          = args.quant,
        code_mode      = True,
        max_new_tokens = args.max_tokens,
    )

    dense_model = None
    if args.baseline:
        print("Loading second copy for rolling-context baseline…")
        from inference.colab_t4 import _MODEL_IDS
        from transformers import AutoModelForCausalLM
        import torch
        dense_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_IDS[args.model],
            torch_dtype   = torch.float16,
            device_map    = "cuda",
            trust_remote_code = True,
        )
        dense_model.eval()

    bench   = MultiTurnBench(engine, dense_model=dense_model,
                             max_new_tokens=args.max_tokens)
    session = CODE_SESSION[: args.turns]
    niah_d  = [] if args.no_niah else _NIAH_DISTANCES

    result  = bench.run(niah_distances=niah_d, verbose=True)
    print_report(result)

    if args.save_csv:
        save_csv(result, args.save_csv)


if __name__ == "__main__":
    main()
