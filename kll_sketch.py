"""
inference/kll_sketch.py
-----------------------
KLL Quantile Sketch adapted for KV-cache eviction scoring.

Extends the base sketch in hnt/quant.py with three additions needed
for live inference eviction:

  1. Exponential time decay — per-step decay factor θ so that old attention
     scores fade, keeping the threshold responsive to recent context windows.

  2. Per-head API — update(scores), step_decay(), threshold(q) — with numpy
     internals hidden behind torch-compatible inputs.

  3. Merge — combine two sketches for cross-head or cross-layer aggregation.

Complexity
----------
  Space  : O(k · log(n/k))  — for sketch capacity k, after n updates
  Update : O(1) amortised   — compaction is O(k) but occurs O(log n/k) times
  Query  : O(k · log k)     — sort + weighted cumsum inside the sketch
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["AttentionKLLSketch"]


class AttentionKLLSketch:
    """
    Streaming weighted quantile estimator (Karnin–Lang–Liberty 2016)
    with exponential time decay for recency-biased KV eviction.

    Parameters
    ----------
    k     : compactor capacity per level — controls ε = O(1/k) error.
            Practical default: 128 gives ±0.8 % quantile error.
    decay : per-step multiplicative weight decay (0 < decay ≤ 1.0).
            0.99 = a 100-step-old score weighs ~37 % of a fresh one.
            1.0  = no decay (classic KLL, stationary distribution).
    """

    def __init__(self, k: int = 128, decay: float = 0.99):
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.k     = k
        self.decay = decay

        # Parallel lists of values and their weights per compactor level
        self._vals: List[List[float]] = [[]]
        self._wts:  List[List[float]] = [[]]
        self.n_seen: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, scores) -> None:
        """Add a batch of attention scores.

        Accepts torch.Tensor, np.ndarray, or any sequence of floats.
        All values are pushed into level-0 with weight 1.0 and compacted
        upward as needed.
        """
        if hasattr(scores, "detach"):           # torch.Tensor
            arr = scores.detach().float().cpu().numpy().ravel()
        else:
            arr = np.asarray(scores, dtype=np.float32).ravel()

        n = len(arr)
        if n == 0:
            return
        self.n_seen += n
        self._vals[0].extend(arr.tolist())
        self._wts[0].extend([1.0] * n)
        self._compact(0)

    def step_decay(self) -> None:
        """Apply one step of weight decay to every stored item.

        Call once per auto-regressive generation step so that the
        eviction threshold adapts to the most recent attention patterns.
        Complexity: O(total stored items) — negligible vs KV pool ops.
        """
        if self.decay >= 1.0:
            return
        d = self.decay
        for lvl in range(len(self._vals)):
            wts = self._wts[lvl]
            for i in range(len(wts)):
                wts[i] *= d

    def threshold(self, quantile: float) -> float:
        """Return the score value at `quantile` fraction of the weight mass.

        Tokens whose cumulative attention score is below threshold(0.15)
        are bottom-15 % candidates for eviction from the important pool.

        Returns 0.0 when the sketch is empty.
        """
        vals, wts = [], []
        for lvl in range(len(self._vals)):
            vals.extend(self._vals[lvl])
            wts.extend(self._wts[lvl])

        if not vals:
            return 0.0

        arr = np.array(vals, dtype=np.float32)
        w   = np.array(wts,  dtype=np.float64)
        idx = np.argsort(arr)
        arr, w = arr[idx], w[idx]
        cum   = np.cumsum(w)
        total = cum[-1]
        if total < 1e-12:
            return float(arr[0])
        pos = int(np.searchsorted(cum, float(quantile) * total))
        return float(arr[min(pos, len(arr) - 1)])

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._vals  = [[]]
        self._wts   = [[]]
        self.n_seen = 0

    def merge(self, other: "AttentionKLLSketch") -> None:
        """In-place merge of another sketch into self.

        Used to aggregate scores from multiple heads or layers before
        computing a single eviction threshold.  O(k) per level.
        """
        for lvl, (v, w) in enumerate(zip(other._vals, other._wts)):
            # Dump everything into level-0 and let compaction sort it out
            self._vals[0].extend(v)
            self._wts[0].extend(w)
        self._compact(0)

    # ── Internal compaction ───────────────────────────────────────────────────

    def _compact(self, lvl: int) -> None:
        """KLL compaction: sort level, keep every other item with doubled weight.

        Triggered when a level overflows its k-item capacity.
        Items are sorted, then a random parity {0, 1} selects which half
        is promoted to the next level (the other is dropped).
        This preserves quantile correctness in expectation.
        """
        if len(self._vals[lvl]) < self.k:
            return

        pairs = sorted(zip(self._vals[lvl], self._wts[lvl]))
        vs    = [p[0] for p in pairs]
        ws    = [p[1] for p in pairs]
        start = int(np.random.randint(0, 2))
        pv    = vs[start::2]
        pw    = [x * 2.0 for x in ws[start::2]]

        self._vals[lvl] = []
        self._wts[lvl]  = []

        next_lvl = lvl + 1
        if next_lvl >= len(self._vals):
            self._vals.append([])
            self._wts.append([])

        self._vals[next_lvl].extend(pv)
        self._wts[next_lvl].extend(pw)
        self._compact(next_lvl)
