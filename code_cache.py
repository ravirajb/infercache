"""
inference/code_cache.py
-----------------------
Code-domain-specific cache optimizations.

Why code is structurally different from prose
---------------------------------------------
Prose:    importance decays with distance.  Old context is rarely referenced.
Code:     importance is NON-MONOTONE.

  Line 1:   import numpy as np          ← still referenced at line 400
  Line 5:   class Transformer:          ← all methods live inside this scope
  Line 10:  def forward(self, x):       ← every call-site needs the signature
  Line 200: return self.attention(x)    ← references 'attention' from line 15
  Line 400: loss = F.cross_entropy(out) ← 'F' from the import on line 1

Default HierarchicalKVCache settings (sink=4, recent=256, important=252) were
tuned for dialogue / QA where importance IS roughly monotone.  Code needs:

  1.  Larger L0 (sink=16)    — preserve import block + first N function headers
  2.  Larger L1 (recent=512) — function bodies span hundreds of lines
  3.  Larger L2 (important=512) — long-range identifier back-references
  4.  Slower decay (0.999 vs 0.99) — old code is just as relevant as new code
  5.  Lower eviction quantile (0.05 vs 0.15) — evict less aggressively
  6.  Earlier two-pass threshold (256 vs 512) — start compression sooner
      because code prompts tend to be longer per "idea" than prose

  Total budget = 16 + 512 + 512 = 1040 tokens.
  For Qwen2.5-3B (36L, 8 kv-heads, 128 dim, fp16):
    Memory ≈ 36 × 8 × 1040 × 128 × 2 bytes ≈ 60 MB per domain cache.

Structural token boosting
-------------------------
CodeTokenScorer decodes each token, checks it against language-specific
keyword sets, and multiplies the attention-derived importance score by a
structural multiplier.  Structural tokens are then less likely to be evicted
from L2.

  Tier-1 (3.0×): def / class / import / from / return / yield / raise /
                  async / await / struct / fn / pub / SELECT / CREATE …
  Tier-2 (2.0×): if / elif / else / for / while / try / except / finally /
                  match / case / with / switch / do / break / continue
  Tier-3 (1.5×): opening brackets { ( [ — scope entry markers
  Tier-4 (1.2×): = operator and closing brackets ) ] } — assignment / scope exit

Identifier pinning
------------------
IdentifierPinner counts token occurrences over a sliding window of history.
Any token (non-keyword, length ≥ 2) that appears ≥ pin_threshold times is
"pinned" — its importance is set to float('inf') so it survives all evictions.

This keeps variable names, function names, and class names that appear
throughout the file in the L2 pool with zero risk of eviction.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from .hierarchical_cache import CacheConfig
from .engine import EngineConfig

__all__ = [
    "CODE_CACHE_CFG",
    "code_engine_config",
    "CodeTokenScorer",
    "IdentifierPinner",
]

# ── Code-tuned cache preset ──────────────────────────────────────────────────

#: Drop-in replacement for a default CacheConfig when the domain is CODE.
CODE_BUDGET_SINK      = 16
CODE_BUDGET_RECENT    = 512
CODE_BUDGET_IMPORTANT = 512


# ── Structural multipliers ────────────────────────────────────────────────────

# Tier 1 — definition / control-flow keywords that define long-range structure
_TIER1_KEYWORDS: frozenset = frozenset({
    # Python
    "def", "class", "import", "from", "return", "yield",
    "raise", "async", "await", "global", "nonlocal",
    # JS / TS
    "function", "const", "let", "var", "export", "default",
    "interface", "type", "enum", "implements", "extends",
    # Rust / Go / C-style
    "fn", "pub", "struct", "impl", "trait", "mod", "use",
    "func", "package", "defer", "go",
    # SQL
    "SELECT", "CREATE", "INSERT", "UPDATE", "DELETE",
    "ALTER", "DROP", "GRANT", "REVOKE",
})
_TIER1_MULT = 3.0

# Tier 2 — flow-control; locally important but not globally defining
_TIER2_KEYWORDS: frozenset = frozenset({
    "if", "elif", "else", "for", "while",
    "try", "except", "finally", "with",
    "match", "case", "switch", "do",
    "break", "continue", "pass",
    # Java / C#
    "throw", "catch", "throws", "instanceof", "new",
    "public", "private", "protected", "static", "final",
    "abstract", "override", "virtual",
    # SQL
    "WHERE", "FROM", "JOIN", "ON", "GROUP", "ORDER",
    "HAVING", "LIMIT", "OFFSET",
})
_TIER2_MULT = 2.0

# Tier 3 — scope-opening brackets (function/block/dict entry markers)
_SCOPE_OPEN  = frozenset({"(", "[", "{", "<"})
_SCOPE_MULT  = 1.5

# Tier 4 — assignment and scope-close (moderately structural)
_ASSIGN_CLOSE = frozenset({"=", ")", "]", "}", "->", "=>", ":"})
_ASSIGN_MULT  = 1.2

# Tokenizer decoration characters that appear as prefix/suffix artifacts
_STRIP_RE = re.compile(r"^[Ġ▁Ā ]+|[Ġ▁Ā ]+$")


class CodeTokenScorer:
    """
    Map token IDs → structural-importance multipliers.

    Usage
    -----
    scorer = CodeTokenScorer(tokenizer)
    # attn_scores: [T] tensor from two-pass summary
    boosted = scorer.boost(input_ids, attn_scores)

    Parameters
    ----------
    tokenizer:
        Any HuggingFace tokenizer.  Must support ``convert_ids_to_tokens()``.
    pin_tracker:
        Optional IdentifierPinner.  If provided, pinned identifiers receive
        infinite importance so they are never evicted from L2.
    """

    def __init__(
        self,
        tokenizer,
        pin_tracker: Optional[IdentifierPinner] = None,
    ) -> None:
        self.tokenizer   = tokenizer
        self.pin_tracker = pin_tracker
        # Cache decoded strings so repeated calls on the same vocab are fast
        self._decoded: Dict[int, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def boost(
        self,
        input_ids:   torch.Tensor,   # [T] int64
        attn_scores: torch.Tensor,   # [T] float32
    ) -> torch.Tensor:
        """
        Return a new score tensor with structural multipliers applied.

        The output is suitable for use as ``importance_scores`` in
        ``HierarchicalKVCache.load_from_hf_output()``.
        """
        ids   = input_ids.tolist()
        mults = self._structural_multipliers(ids)     # List[float], length T
        mult_t = torch.tensor(mults, dtype=torch.float32, device=attn_scores.device)

        boosted = attn_scores.float() * mult_t

        # Pins get infinity so they always beat the eviction threshold
        if self.pin_tracker is not None:
            pinned_mask = self.pin_tracker.pinned_mask(ids, device=attn_scores.device)
            boosted = torch.where(pinned_mask, torch.full_like(boosted, float("inf")), boosted)

        return boosted

    def warm_up(self, vocab_size: int = 32_000) -> None:
        """Pre-decode the most common token IDs for faster inference.  Optional."""
        ids = list(range(min(vocab_size, 32_000)))
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        for tid, tok in zip(ids, tokens):
            self._decoded[tid] = _STRIP_RE.sub("", tok) if tok else ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _decode(self, tid: int) -> str:
        if tid in self._decoded:
            return self._decoded[tid]
        tok = self.tokenizer.convert_ids_to_tokens([tid])[0] or ""
        s = _STRIP_RE.sub("", tok)
        self._decoded[tid] = s
        return s

    def _structural_multipliers(self, ids: List[int]) -> List[float]:
        mults: List[float] = []
        for tid in ids:
            s = self._decode(tid)
            if s in _TIER1_KEYWORDS:
                mults.append(_TIER1_MULT)
            elif s in _TIER2_KEYWORDS:
                mults.append(_TIER2_MULT)
            elif s in _SCOPE_OPEN:
                mults.append(_SCOPE_MULT)
            elif s in _ASSIGN_CLOSE:
                mults.append(_ASSIGN_MULT)
            else:
                mults.append(1.0)
        return mults


class IdentifierPinner:
    """
    Track token occurrences; pin tokens that appear frequently.

    "Pinned" tokens get importance = inf, guaranteeing they stay in L2
    for the lifetime of the cache session.

    Parameters
    ----------
    pin_threshold:
        Number of appearances required before a token is pinned (default: 3).
    min_token_len:
        Minimum decoded length to be considered an identifier (filters
        punctuation, single-character operators, etc.).
    window:
        How many tokens to count before resetting the counter.
        None = never reset (count over entire session).
    """

    def __init__(
        self,
        tokenizer,
        pin_threshold: int = 3,
        min_token_len: int = 2,
        window: Optional[int] = None,
    ) -> None:
        self.tokenizer     = tokenizer
        self.pin_threshold = pin_threshold
        self.min_len       = min_token_len
        self.window        = window

        self._counter: Counter = Counter()
        self._pinned:  Set[int] = set()
        self._history: List[int] = []
        self._decoded: Dict[int, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, input_ids: torch.Tensor) -> None:
        """Ingest a new sequence of token IDs and update pin decisions."""
        ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
        self._history.extend(ids)

        # Sliding window: drop old counts
        if self.window and len(self._history) > self.window:
            removed = self._history[: len(self._history) - self.window]
            for tid in removed:
                self._counter[tid] = max(0, self._counter[tid] - 1)
            self._history = self._history[-self.window :]

        for tid in ids:
            tok = self._decode(tid)
            if not self._is_candidate(tok):
                continue
            self._counter[tid] += 1
            if self._counter[tid] >= self.pin_threshold:
                self._pinned.add(tid)

    def pinned_mask(
        self,
        ids: List[int],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Return bool [T] tensor: True where token should be pinned."""
        mask = [tid in self._pinned for tid in ids]
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def reset(self) -> None:
        self._counter.clear()
        self._pinned.clear()
        self._history.clear()

    @property
    def n_pinned_types(self) -> int:
        """Number of distinct token types currently pinned."""
        return len(self._pinned)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _decode(self, tid: int) -> str:
        if tid in self._decoded:
            return self._decoded[tid]
        tok = self.tokenizer.convert_ids_to_tokens([tid])[0] or ""
        s = _STRIP_RE.sub("", tok)
        self._decoded[tid] = s
        return s

    def _is_candidate(self, tok: str) -> bool:
        if len(tok) < self.min_len:
            return False
        if tok in _TIER1_KEYWORDS or tok in _TIER2_KEYWORDS:
            return False          # keywords handled by scorer, not pinner
        if not re.search(r"[a-zA-Z_]", tok):
            return False          # pure punctuation / numbers
        return True


# ── Factory: code-tuned EngineConfig ─────────────────────────────────────────

def code_engine_config(
    base_cfg: EngineConfig,
    *,
    max_new_tokens: int  = 512,
    temperature:    float = 0.2,   # low temp — code should be deterministic
    top_p:          float = 0.95,
    repetition_penalty: float = 1.1,  # mild; prevents repeated boilerplate
) -> EngineConfig:
    """
    Return a copy of *base_cfg* with code-optimised parameters applied.

    Typically called after ``build_qwen_config(model_name)``::

        from inference import build_qwen_config
        from inference.code_cache import code_engine_config

        base   = build_qwen_config("Qwen/Qwen2.5-3B-Instruct")
        cfg    = code_engine_config(base)
        engine = HierarchicalInferenceEngine.from_pretrained(model_name, cfg)

    Changes applied
    ---------------
    sink_size:          4   → 16      (preserve import block)
    recent_size:       256  → 512     (function body local context)
    important_size:    252  → 512     (long-range identifier references)
    cache_decay:       0.99 → 0.999   (old code is still relevant)
    evict_quantile:    0.15 → 0.05    (evict very conservatively)
    summary_threshold: 512  → 256     (compress sooner; code prompts are dense)
    max_new_tokens:    256  → 512     (code outputs are longer)
    temperature:       1.0  → 0.2     (code should be mostly deterministic)
    repetition_penalty:1.0  → 1.1     (mild penalty against repeated boilerplate)
    """
    import dataclasses
    return dataclasses.replace(
        base_cfg,
        # cache budget
        sink_size           = CODE_BUDGET_SINK,
        recent_size         = CODE_BUDGET_RECENT,
        important_size      = CODE_BUDGET_IMPORTANT,
        # eviction behaviour
        cache_decay         = 0.999,
        evict_quantile      = 0.05,
        # generation: start two-pass earlier; code prompts are dense
        summary_threshold   = 256,
        # generation quality
        max_new_tokens      = max_new_tokens,
        temperature         = temperature,
        top_p               = top_p,
        repetition_penalty  = repetition_penalty,
    )
