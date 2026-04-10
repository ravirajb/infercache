"""
inference/hierarchical_cache.py
--------------------------------
Hierarchical KV cache with a fixed-size budget and KLL-guided eviction.

Slot layout (all sizes are tokens, not bytes)
----------------------------------------------

  ┌──────────────────────────────────────────────────────┐
  │  L0 sink  │  L1 recent  │  L2 important              │
  │ sink_size │ recent_size │   important_size            │
  └──────────────────────────────────────────────────────┘
         ↑            ↑                    ↑
    Never evicted   FIFO ring      KLL-scored, evictable

  total_budget = sink_size + recent_size + important_size
               = constant regardless of conversation length

Level semantics
---------------
L0 — Attention sinks
  First `sink_size` token positions (typically 4–8).
  StreamingLLM showed that LLMs route a disproportionate fraction of
  attention mass to these initial tokens regardless of content.
  Dropping them degrades even unrelated generation.  NEVER evicted.

L1 — Recent sliding window
  Last `recent_size` token positions, FIFO ring.
  Ensures local coherence for the current generation burst.
  Tokens here are not scored; they are always retained.
  When the ring overflows, the displaced token is scored and
  considered for promotion to L2 before being dropped.

L2 — Important token pool
  Up to `important_size` tokens, scored by cumulative attention mass.
  Each token carries a scalar importance score that is updated every
  time it is attended to.  When L2 is full and a new token arrives,
  the slot with the minimum importance score is evicted (or the new
  token is discarded if it scores below the KLL eviction threshold).

Complexity per generation step
-------------------------------
  Lookup   :  O(budget)  — fixed, independent of context length  → "O(1)"
  Update   :  O(budget)  — KLL sketch update + eviction check
  Memory   :  O(num_layers × num_kv_heads × budget × head_dim × sizeof(dtype))
              For Qwen 3B (28L, 8 kv-heads, 128 dim, fp16, budget=512):
              ≈ 28 × 8 × 512 × 128 × 2 = 29 MB (negligible)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import numpy as np

from .kll_sketch   import AttentionKLLSketch
from .cache_safety import normalise_kv as _cs_normalise_kv

__all__ = ["CacheConfig", "CacheStats", "HierarchicalKVCache", "_normalise_past_kv"]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CacheConfig:
    num_layers:     int
    num_kv_heads:   int
    head_dim:       int

    sink_size:      int   = 4      # L0 — attention-sink tokens (never evicted)
    recent_size:    int   = 256    # L1 — sliding-window recent tokens
    important_size: int   = 252    # L2 — KLL-scored important tokens

    evict_quantile: float = 0.15   # bottom q-fraction of L2 is eviction candidate
    decay:          float = 0.99   # per-step time decay for importance scores
    kll_k:          int   = 128    # KLL compactor capacity per head

    dtype: torch.dtype = torch.float16

    @property
    def total_budget(self) -> int:
        return self.sink_size + self.recent_size + self.important_size

    @property
    def recent_start(self) -> int:
        return self.sink_size

    @property
    def important_start(self) -> int:
        return self.sink_size + self.recent_size


# ── Statistics ────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    total_tokens_seen: int = 0
    total_evictions:   int = 0
    current_fill:      int = 0
    sink_fill:         int = 0
    recent_fill:       int = 0
    important_fill:    int = 0


# ── Main cache class ──────────────────────────────────────────────────────────

class HierarchicalKVCache:
    """
    Fixed-budget, three-level KV cache with KLL-driven eviction.

    All tensor buffers are pre-allocated at construction time.
    No dynamic allocation occurs during generation.

    Thread-safety: none — designed for single-threaded generation.
    """

    def __init__(self, cfg: CacheConfig, device=None):
        self.cfg    = cfg
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        budget = cfg.total_budget

        # ── KV tensor buffers (preallocated) ──────────────────────────────
        # Shape: [num_layers, num_kv_heads, budget, head_dim]
        shape = (cfg.num_layers, cfg.num_kv_heads, budget, cfg.head_dim)
        self.K = torch.zeros(shape, dtype=cfg.dtype, device=self.device)
        self.V = torch.zeros(shape, dtype=cfg.dtype, device=self.device)

        # ── Slot metadata (CPU) ───────────────────────────────────────────
        # sequence position of token in slot (-1 = empty)
        self.positions:  torch.Tensor = torch.full((budget,), -1, dtype=torch.long)
        # accumulated importance score (attention mass)
        self.importance: torch.Tensor = torch.zeros(budget, dtype=torch.float32)
        # True for filled slots
        self.occupied:   torch.Tensor = torch.zeros(budget, dtype=torch.bool)

        # ── Fill pointers ─────────────────────────────────────────────────
        self._sink_fill:      int = 0   # [0, sink_size)
        self._recent_fill:    int = 0   # [0, recent_size)
        self._recent_ptr:     int = cfg.recent_start   # write head (ring buffer)
        self._important_fill: int = 0   # [0, important_size)

        # ── Per-head KLL sketches for L2 ──────────────────────────────────
        # [num_layers][num_kv_heads]  — one sketch per head per layer
        self._sketches: List[List[AttentionKLLSketch]] = [
            [
                AttentionKLLSketch(k=cfg.kll_k, decay=cfg.decay)
                for _ in range(cfg.num_kv_heads)
            ]
            for _ in range(cfg.num_layers)
        ]

        self._stats = CacheStats()
        self._step  = 0

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def total_budget(self) -> int:
        return self.cfg.total_budget

    @property
    def current_fill(self) -> int:
        return self._sink_fill + self._recent_fill + self._important_fill

    @property
    def stats(self) -> CacheStats:
        s = self._stats
        s.current_fill   = self.current_fill
        s.sink_fill      = self._sink_fill
        s.recent_fill    = self._recent_fill
        s.important_fill = self._important_fill
        return s

    # ── Bulk load from a HuggingFace summary-pass output ─────────────────────

    def load_from_hf_output(
        self,
        past_key_values,                    # HF past_key_values tuple
        importance_scores: torch.Tensor,    # [seq_len]  — higher = more important
        positions:         Optional[torch.Tensor] = None,  # [seq_len]
    ) -> None:
        """
        Populate the cache from a HuggingFace model's past_key_values.
        Called after the summary pass to prime all three levels.

        past_key_values layout
        ----------------------
        Tuple of num_layers elements; each element is (K, V) where
        K.shape = V.shape = [batch=1, num_kv_heads, seq_len, head_dim].

        Supports both:
          - Tuple-of-tuples (legacy transformers < 4.36)
          - DynamicCache /  StaticCache (transformers ≥ 4.36)
        """
        self.reset()

        # Normalise to tuple-of-tuples regardless of cache type
        pkv = _normalise_past_kv(past_key_values)
        seq_len = pkv[0][0].shape[2]

        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.long)

        scores = importance_scores.float().cpu()

        for tok in range(seq_len):
            pos   = int(positions[tok].item())
            score = float(scores[tok].item())

            # L0: sink — keep first sink_size absolute positions
            if pos < self.cfg.sink_size and self._sink_fill < self.cfg.sink_size:
                slot = self._sink_fill
                self._write_kv(slot, pkv, tok)
                self.positions[slot]  = pos
                self.importance[slot] = score
                self.occupied[slot]   = True
                self._sink_fill += 1
                continue

            # L1: recent — last recent_size tokens (tail of the sequence)
            if (seq_len - tok) <= self.cfg.recent_size:
                slot = self.cfg.recent_start + self._recent_fill
                if self._recent_fill < self.cfg.recent_size:
                    self._write_kv(slot, pkv, tok)
                    self.positions[slot]  = pos
                    self.importance[slot] = score
                    self.occupied[slot]   = True
                    self._recent_fill += 1
                continue

            # L2: important — elect by score, evict if full
            self._try_insert_important(pkv, tok, score, pos)

        self._stats.total_tokens_seen += seq_len

    # ── Live push during auto-regressive generation ───────────────────────────

    def push(
        self,
        key_states:   torch.Tensor,             # [num_layers, num_kv_heads, 1, head_dim]
        value_states: torch.Tensor,             # same
        position:     int,
        attn_weights: Optional[torch.Tensor] = None,  # [num_layers, num_kv_heads, 1, F]
    ) -> None:
        """
        Ingest the KV of a newly generated token.

        1. Boost L2 importance scores using current attention weights.
        2. Place new token in L1 ring, promoting displaced token to L2
           if it's important enough.
        3. Apply per-step decay to all KLL sketches.

        Cost: O(budget) — constant in context length.
        """
        self._step += 1

        # 1. Update importance from attention weights
        if attn_weights is not None:
            self._boost_importance(attn_weights)

        # 2. The slot the ring is about to overwrite
        slot = self._recent_ptr
        if self.occupied[slot]:
            # Displaced token: give it a chance to survive in L2
            self._maybe_promote_to_l2(slot)

        # Write new token into ring slot
        for li in range(self.cfg.num_layers):
            self.K[li, :, slot, :] = key_states[li, :, 0, :].to(
                self.device, self.cfg.dtype)
            self.V[li, :, slot, :] = value_states[li, :, 0, :].to(
                self.device, self.cfg.dtype)
        self.positions[slot]  = position
        self.importance[slot] = 0.0   # fresh — no attention accumulated yet
        self.occupied[slot]   = True

        self._recent_fill = min(self._recent_fill + 1, self.cfg.recent_size)
        # Advance ring pointer
        rs = self.cfg.recent_size
        self._recent_ptr = (
            self.cfg.recent_start
            + (self._recent_ptr - self.cfg.recent_start + 1) % rs
        )

        self._stats.total_tokens_seen += 1

        # 3. Time decay
        for layer_sketches in self._sketches:
            for sk in layer_sketches:
                sk.step_decay()

    # ── Export for HuggingFace attention ──────────────────────────────────────

    def to_hf_past_key_values(
        self,
        device: Optional[torch.device] = None,
    ):
        """
        Export the filled cache as HuggingFace past_key_values.

        Returns a tuple (one element per layer) of (K, V) pairs:
          K.shape = V.shape = [1, num_kv_heads, F, head_dim]
        where F = number of currently occupied slots, sorted by position
        so that causal attention is correct.

        RoPE note: K/V vectors were already position-rotated by the model
        during the summary pass.  Passing them back as-is is correct —
        the attention for the current query token will see the right
        key–value semantics at their original positions.
        Returns None if the cache is empty.
        """
        dev = device or self.device
        fill_mask = self.occupied               # [budget] bool
        if not fill_mask.any():
            return None

        occupied_slots = fill_mask.nonzero(as_tuple=True)[0]  # [F]
        occupied_pos   = self.positions[occupied_slots]        # [F] positions
        sort_order     = occupied_pos.argsort()
        sorted_slots   = occupied_slots[sort_order]            # [F] sorted by pos

        # Build per-layer K/V tensors.
        layers_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for li in range(self.cfg.num_layers):
            K_li = self.K[li][:, sorted_slots, :].unsqueeze(0).to(dev)  # [1, H, F, D]
            V_li = self.V[li][:, sorted_slots, :].unsqueeze(0).to(dev)
            layers_kv.append((K_li, V_li))

        # transformers >= 4.36 passes past_key_values straight to
        # cache.get_seq_length(), so the model rejects a plain tuple.
        # Build a DynamicCache and pre-populate each layer directly.
        try:
            from transformers.cache_utils import DynamicCache, DynamicLayer
            cache = DynamicCache()
            for K_li, V_li in layers_kv:
                layer = DynamicLayer()
                layer.dtype        = K_li.dtype
                layer.device       = K_li.device
                layer.keys         = K_li
                layer.values       = V_li
                layer.is_initialized = True
                cache.layers.append(layer)
            return cache
        except ImportError:
            pass

        # Legacy fallback for transformers < 4.36.
        return tuple(layers_kv)

    def get_position_ids_for_cache(self) -> torch.Tensor:
        """
        Return sorted sequence positions of all cached tokens.
        Useful for constructing `position_ids` when querying the cache.
        """
        occ   = self.occupied.nonzero(as_tuple=True)[0]
        pos   = self.positions[occ]
        return pos.sort().values

    def reset(self) -> None:
        """Clear all state.  O(budget) — fills preallocated zeros."""
        self.K.zero_()
        self.V.zero_()
        self.positions.fill_(-1)
        self.importance.zero_()
        self.occupied.zero_()

        self._sink_fill      = 0
        self._recent_fill    = 0
        self._recent_ptr     = self.cfg.recent_start
        self._important_fill = 0
        self._step           = 0

        for layer_sketches in self._sketches:
            for sk in layer_sketches:
                sk.reset()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _write_kv(self, slot: int, pkv, tok_idx: int) -> None:
        """Copy K/V from pkv[:][batch=0, :, tok_idx, :] into slot."""
        for li in range(min(self.cfg.num_layers, len(pkv))):
            self.K[li, :, slot, :] = pkv[li][0][0, :, tok_idx, :].to(
                self.device, self.cfg.dtype)
            self.V[li, :, slot, :] = pkv[li][1][0, :, tok_idx, :].to(
                self.device, self.cfg.dtype)

    def _try_insert_important(self, pkv, tok_idx: int, score: float, pos: int) -> None:
        """Insert a token into L2; evict the lowest-scoring incumbent if full."""
        start    = self.cfg.important_start
        capacity = self.cfg.important_size

        if self._important_fill < capacity:
            slot = start + self._important_fill
            self._write_kv(slot, pkv, tok_idx)
            self.positions[slot]  = pos
            self.importance[slot] = score
            self.occupied[slot]   = True
            self._important_fill += 1
            # Feed score into per-head KLL sketches
            for layer_sks in self._sketches:
                for sk in layer_sks:
                    sk.update([score])
        else:
            # Evict the slot with the minimum importance in L2
            evict_slot = self._l2_min_slot()
            evict_score = float(self.importance[evict_slot].item())
            if score > evict_score:
                self._write_kv(slot=evict_slot, pkv=pkv, tok_idx=tok_idx)
                self.positions[evict_slot]  = pos
                self.importance[evict_slot] = score
                self.occupied[evict_slot]   = True
                self._stats.total_evictions += 1
                for layer_sks in self._sketches:
                    for sk in layer_sks:
                        sk.update([score])

    def _l2_min_slot(self) -> int:
        """Return the L2 slot index with the lowest importance score."""
        start = self.cfg.important_start
        end   = start + self._important_fill
        local_min = int(self.importance[start:end].argmin().item())
        return start + local_min

    def _boost_importance(self, attn_weights: torch.Tensor) -> None:
        """Accumulate attention mass from attn_weights into slot importance.

        attn_weights: [num_layers, num_kv_heads, 1, num_cached_tokens]
        or            [num_layers, num_kv_heads, num_cached_tokens]
        The caller must ensure the last dim aligns with occupied slots.
        """
        # Collapse to [n_cached] by averaging over layers, heads, queries
        if attn_weights.dim() == 4:
            agg = attn_weights.float().mean(dim=(0, 1, 2))   # [F]
        else:
            agg = attn_weights.float().mean(dim=(0, 1))       # [F]

        # Map to slot indices in importance-score order (same as to_hf output)
        occ           = self.occupied.nonzero(as_tuple=True)[0]
        sort_order    = self.positions[occ].argsort()
        sorted_slots  = occ[sort_order]

        n = min(agg.shape[0], sorted_slots.shape[0])
        self.importance[sorted_slots[:n]] += agg[:n].cpu()

    def _maybe_promote_to_l2(self, recent_slot: int) -> None:
        """When a recent-window token is about to be overwritten, check if
        it has accumulated enough importance to claim a spot in L2."""
        score = float(self.importance[recent_slot].item())
        if score <= 0.0:
            return

        start    = self.cfg.important_start
        capacity = self.cfg.important_size

        if self._important_fill < capacity:
            # Free spot: copy this token to L2
            target_slot = start + self._important_fill
            self.K[:, :, target_slot, :] = self.K[:, :, recent_slot, :]
            self.V[:, :, target_slot, :] = self.V[:, :, recent_slot, :]
            self.positions[target_slot]  = self.positions[recent_slot].item()
            self.importance[target_slot] = score
            self.occupied[target_slot]   = True
            self._important_fill += 1
        else:
            # Evict L2 minimum if this token is more important
            evict_slot  = self._l2_min_slot()
            evict_score = float(self.importance[evict_slot].item())
            if score > evict_score:
                self.K[:, :, evict_slot, :] = self.K[:, :, recent_slot, :]
                self.V[:, :, evict_slot, :] = self.V[:, :, recent_slot, :]
                self.positions[evict_slot]  = self.positions[recent_slot].item()
                self.importance[evict_slot] = score
                self.occupied[evict_slot]   = True
                self._stats.total_evictions += 1


# ── Helper: normalise HF past_key_values to tuple-of-tuples ──────────────────

def _normalise_past_kv(past_key_values):
    """
    Thin wrapper around cache_safety.normalise_kv.

    Kept here for backward-compatibility (two_pass.py imports it from this
    module).  All normalisation logic lives in cache_safety.py so it is
    tested and updated in one place.
    """
    return _cs_normalise_kv(past_key_values)
