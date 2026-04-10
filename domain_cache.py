"""
inference/domain_cache.py
--------------------------
Multiple domain-specific hierarchical KV caches.

Motivation
----------
Different contexts leave very different "important token" footprints:
  • Code generation  — keywords, brackets, identifiers, type signatures.
  • Q&A / reasoning  — question subjects, fact tokens, answer anchors.
  • General language — discourse connectives, named entities, topic terms.

A single shared cache must evict tokens from unrelated domains to make room,
destroying cross-domain coherence.  Domain-specific caches let code tokens
persist without competing against Q&A fact tokens.

Architecture
------------
  MultiDomainCache
    ├── DomainRouter          lightweight heuristic classifier
    │     (no model inference — pure token statistics)
    ├── HierarchicalKVCache   [CODE]
    ├── HierarchicalKVCache   [QA]
    └── HierarchicalKVCache   [LANGUAGE]

Budget per domain:
  Each domain cache uses the full CacheConfig budget independently.
  Memory cost = N_domains × budget × layers × kv_heads × head_dim × dtype_bytes.
  For 3 domains, Qwen 3B, budget=512 → 3 × 29 MB ≈ 87 MB extra.

Routing rules (heuristic — no model call required):
  CODE     : first 200 tokens contain ≥ 3 % code-like characters   OR
             explicit code-block marker (```/~~~) detected           OR
             common keyword ratio > threshold
  QA       : prompt starts with a wh-word (what/why/how/when/where/who)
             OR ends with "?"  OR contains "explain"/"define"
  LANGUAGE : default bucket

Router accuracy is intentionally approximate — incorrect routing routes to
a suboptimal cache, not incorrect generation.  Worst case = performance
regression back to a cold cache, not wrong output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import torch

from .hierarchical_cache import CacheConfig, HierarchicalKVCache

__all__ = ["Domain", "DomainRouter", "MultiDomainCache"]


# ── Domain enum ───────────────────────────────────────────────────────────────

class Domain(Enum):
    CODE     = auto()
    QA       = auto()
    LANGUAGE = auto()


# ── Routing heuristics ────────────────────────────────────────────────────────

# Code: characters that occur frequently in code but rarely in prose
_CODE_CHARS   = frozenset("{}[]()<>=!&|^~;:#@$\\")
_CODE_THRESH  = 0.03      # fraction of chars that are code-like

_CODE_TOKENS  = frozenset([
    "def", "class", "import", "return", "if", "else", "elif", "for",
    "while", "function", "const", "let", "var", "fn", "pub", "struct",
    "interface", "async", "await", "try", "catch", "throw", "raise",
    "print", "printf", "echo", "SELECT", "FROM", "WHERE", "INSERT",
    "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
])
_CODE_KW_THRESH = 2   # at least this many code keywords in the prompt

# Q&A: wh-words and question starters
_QA_STARTERS  = re.compile(
    r"^\s*(what|why|how|when|where|who|which|can you|could you|please explain"
    r"|explain|describe|summarise|summarize|define|tell me|list)",
    re.IGNORECASE,
)
_QA_END = re.compile(r"\?\s*$")


class DomainRouter:
    """
    Lightweight prompt-to-domain classifier.

    All decisions are made on raw text (no model call).
    Intended to run in < 0.1 ms even for 2K-token prompts.
    """

    def route(self, text: str) -> Domain:
        """Classify `text` into a domain.  Returns the most likely Domain."""
        # ── Code detection (highest priority) ─────────────────────────────
        if self._is_code(text):
            return Domain.CODE

        # ── Q&A detection ─────────────────────────────────────────────────
        if self._is_qa(text):
            return Domain.QA

        # ── Default ───────────────────────────────────────────────────────
        return Domain.LANGUAGE

    def _is_code(self, text: str) -> bool:
        # Explicit code fence marker
        if "```" in text or "~~~" in text:
            return True

        # Code-character density
        sample = text[:2000]   # only look at first 2000 chars for speed
        if len(sample) > 0:
            code_ratio = sum(1 for c in sample if c in _CODE_CHARS) / len(sample)
            if code_ratio >= _CODE_THRESH:
                return True

        # Keyword count in first 300 whitespace-separated tokens
        words = text.split()[:300]
        kw_count = sum(1 for w in words if w.strip("():,;.!?") in _CODE_TOKENS)
        if kw_count >= _CODE_KW_THRESH:
            return True

        return False

    def _is_qa(self, text: str) -> bool:
        stripped = text.strip()
        if _QA_STARTERS.match(stripped):
            return True
        if _QA_END.search(stripped):
            return True
        # "explain X", "define X" anywhere near the start
        first_100 = stripped[:100].lower()
        for kw in ("explain", "define", "describe", "summarize", "summarise"):
            if kw in first_100:
                return True
        return False


# ── Multi-domain cache ────────────────────────────────────────────────────────

class MultiDomainCache:
    """
    Three independent hierarchical KV caches, one per domain.

    The `active` domain cache is used for loading / generation.
    Switching domains does NOT clear other domain caches — they persist
    for the lifetime of the session, ready for when that domain is needed again.

    Memory cost: 3 × budget × layers × kv_heads × head_dim × bytes.

    Usage
    -----
    cache = MultiDomainCache(cache_cfg, device)
    domain = cache.route("What is quantum entanglement?")   # → Domain.QA
    active = cache.get(domain)       # → HierarchicalKVCache for QA
    active.load_from_hf_output(...)  # prime with summary pass KV
    hf_pkv = active.to_hf_past_key_values()  # use for generation
    """

    def __init__(
        self,
        cache_cfg: CacheConfig,
        device=None,
        router: Optional[DomainRouter] = None,
    ):
        self._cfg    = cache_cfg
        self._device = device
        self._router = router or DomainRouter()

        self._caches: Dict[Domain, HierarchicalKVCache] = {
            Domain.CODE:     HierarchicalKVCache(cache_cfg, device=device),
            Domain.QA:       HierarchicalKVCache(cache_cfg, device=device),
            Domain.LANGUAGE: HierarchicalKVCache(cache_cfg, device=device),
        }
        self._active_domain: Domain = Domain.LANGUAGE

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, text: str) -> Domain:
        """Classify prompt text and return the recommended Domain."""
        domain = self._router.route(text)
        self._active_domain = domain
        return domain

    @property
    def active_domain(self) -> Domain:
        return self._active_domain

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, domain: Optional[Domain] = None) -> HierarchicalKVCache:
        """Return the HierarchicalKVCache for `domain` (or the active domain)."""
        key = domain if domain is not None else self._active_domain
        return self._caches[key]

    def get_active(self) -> HierarchicalKVCache:
        """Return the cache for the currently active domain."""
        return self._caches[self._active_domain]

    # ── Convenience ───────────────────────────────────────────────────────────

    def reset_domain(self, domain: Domain) -> None:
        """Clear only one domain's cache (e.g., after a topic switch)."""
        self._caches[domain].reset()

    def reset_all(self) -> None:
        """Clear all domain caches."""
        for cache in self._caches.values():
            cache.reset()

    def stats(self) -> Dict[str, object]:
        """Return fill statistics for all domains."""
        return {
            d.name: cache.stats
            for d, cache in self._caches.items()
        }

    # ── Context manager ───────────────────────────────────────────────────────

    def with_domain(self, text: str):
        """Context manager: auto-route text and yield the right cache.

        Usage::
            with multi_cache.with_domain(prompt) as cache:
                cache.load_from_hf_output(...)
                past_kv = cache.to_hf_past_key_values()
        """
        return _DomainContext(self, text)


class _DomainContext:
    def __init__(self, mdc: MultiDomainCache, text: str):
        self._mdc    = mdc
        self._domain = mdc.route(text)

    def __enter__(self) -> HierarchicalKVCache:
        return self._mdc.get(self._domain)

    def __exit__(self, *_):
        pass
