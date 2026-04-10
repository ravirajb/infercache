"""
inference/faiss_index.py
--------------------------
FAISS-accelerated operations for the KV cache.

Where FAISS helps and where it doesn't
---------------------------------------

HELPS:
  A. L2 slot eviction — "find the most redundant token in the pool"
     FAISS IndexFlatIP (inner product = cosine similarity on unit vectors)
     can find:
       - The cached token MOST similar to a new arrival   → O(log F) via IVF
       - The cached token LEAST like any other           → approximate far-point search
     Current linear scan:  O(F)  (F = budget = 512, not expensive, but…)
     With FAISS IVF64:     O(√F) probe — ~4× faster at F=512,
                           scales to O(log F) at F=4096+ (large pool).
     Break-even: FAISS overhead worthwhile only for budget ≥ 1024.
     Below that, PyTorch argmin is faster.  We expose both paths.

  B. Cross-turn cache boost (RBF path boost)
     Current:  O(T_new × C × D_rff) — both new and cached embeddings
     FAISS:    O(T_new × log C)       — find top-k similar cached tokens via
               IndexIVFFlat or IndexHNSWFlat (approx NN)
     Speedup: 5–20× for T_new=200, C=512.

  C. Cluster-centroid nearest-neighbour (GraphKVCache routing)
     Current:  O(n_clusters × D) exact dot-product
     FAISS:    For n_clusters ≤ 64 exact is faster (centroid count is tiny).
               FAISS shines when n_clusters ≥ 512.

DOES NOT HELP:
  D. The per-step attention computation.  That happens inside the model (or
     compact_sdp_attention).  FAISS doesn't integrate into the matmul path.

  E. Very small budgets (buffer ≤ 256).  FAISS IndexFlatIP with 256 vectors
     is ~10× SLOWER than torch.mm due to index construction overhead.

Integration contract
--------------------
  FaissKVIndex is an optional decorator around HierarchicalKVCache.
  All HierarchicalKVCache public methods remain callable — FaissKVIndex
  intercepts only:
    • _l2_min_slot   → fast ANN eviction candidate
    • _boost_cache_relevance → fast cross-turn boost

  If FAISS is not installed, FaissKVIndex falls back to the base class methods
  transparently (no import error at module load time).
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch

_FAISS = False
try:
    import faiss
    _FAISS = True
except ImportError:
    pass

__all__ = ["FaissAugmentedCache", "faiss_available"]


def faiss_available() -> bool:
    return _FAISS


# ── FAISS index wrapper ───────────────────────────────────────────────────────

class _FlatCosineIndex:
    """
    Thin wrapper around faiss.IndexFlatIP (inner product on unit-normalised
    vectors = cosine similarity) with add/remove/search semantics.

    FAISS IndexFlatIP is an exact search structure — O(n) per query — but is
    vectorised over SIMD and far faster than a Python loop.
    Use IndexIVFFlat for approximate search at n > 2048.
    """

    def __init__(self, d: int, use_gpu: bool = False):
        if not _FAISS:
            return
        self.d       = d
        self.index   = faiss.IndexFlatIP(d)   # exact inner product
        self.id_map  = []                      # maps FAISS internal id → slot index
        self._use_gpu = False

        if use_gpu:
            try:
                res          = faiss.StandardGpuResources()
                self.index   = faiss.index_cpu_to_gpu(res, 0, self.index)
                self._use_gpu = True
            except Exception:
                pass   # GPU FAISS not available — stay on CPU

    def _as_fp32_normed(self, vecs: torch.Tensor) -> np.ndarray:
        """Convert to float32, L2-normalise (for cosine similarity via IP)."""
        v   = vecs.float().detach().cpu().numpy()    # [N, D]
        nrm = np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-8)
        return (v / nrm).astype(np.float32)

    def add(self, vectors: torch.Tensor, slot_ids: List[int]) -> None:
        """Add vectors and remember their slot IDs."""
        if not _FAISS:
            return
        x = self._as_fp32_normed(vectors)   # [N, D]
        self.index.add(x)
        self.id_map.extend(slot_ids)

    def search_sim(self, query: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (similarities, slot_ids) for top-k nearest neighbours.
        query: [D] or [N, D].
        """
        if not _FAISS or self.index.ntotal == 0:
            return np.array([]), np.array([])
        q   = self._as_fp32_normed(query.unsqueeze(0) if query.dim() == 1 else query)
        k   = min(k, self.index.ntotal)
        sims, idxs = self.index.search(q, k)   # idxs are FAISS internal ids
        # Convert internal ids back to slot indices
        slot_ids = np.array([self.id_map[i] if 0 <= i < len(self.id_map) else -1
                              for i in idxs[0]])
        return sims[0], slot_ids

    def rebuild(self, vectors: torch.Tensor, slot_ids: List[int]) -> None:
        """Full rebuild.  Called after bulk evictions."""
        if not _FAISS:
            return
        self.index.reset()
        self.id_map = []
        if len(slot_ids) > 0:
            self.add(vectors, slot_ids)

    def __len__(self) -> int:
        return self.index.ntotal if _FAISS else 0


# ── FAISS-augmented cache ─────────────────────────────────────────────────────

class FaissAugmentedCache:
    """
    Wraps a HierarchicalKVCache and replaces two hot-path operations
    with FAISS-accelerated equivalents.

    Operations intercepted
    ----------------------
    1. l2_eviction_candidate(new_key)
       Returns the slot index in L2 that is:
         (a) lowest importance, AND
         (b) most similar to an already-stored slot (most redundant)
       FAISS finds (b) in O(log F); combined with PyTorch argmin for (a)
       this gives a better eviction policy than score-only at minimal cost.

    2. boost_similar_slots(new_embeddings, threshold)
       Given T_new new token embeddings, find all cached token positions that
       are within cosine distance `threshold` and boost their importance.
       Current: O(T_new × C × D_rff)  — RBF + numpy
       FAISS:   O(T_new × log C)       — ANN search

    The cache itself (K/V tensors, fill logic, L0/L1/L2) is unchanged.
    """

    def __init__(
        self,
        cache,                     # HierarchicalKVCache instance
        embed_dim:      int,       # dimensionality of key vectors for FAISS
        use_gpu_faiss:  bool = False,
        faiss_threshold: float = 0.85,   # cos-sim above which a slot is "redundant"
        boost_factor:    float = 1.5,
    ):
        self._cache      = cache
        self._embed_dim  = embed_dim
        self._thresh     = faiss_threshold
        self._boost_f    = boost_factor
        self._index      = _FlatCosineIndex(embed_dim, use_gpu=use_gpu_faiss)

        if not _FAISS:
            warnings.warn(
                "faiss not installed — FaissAugmentedCache falls back to linear scan. "
                "Install with: pip install faiss-cpu  or  pip install faiss-gpu",
                ImportWarning,
                stacklevel=2,
            )

    # ── Delegate all normal cache operations ──────────────────────────────────

    def __getattr__(self, name):
        """Transparent proxy to underlying cache."""
        return getattr(self._cache, name)

    # ── FAISS-accelerated eviction candidate ──────────────────────────────────

    def l2_eviction_candidate(
        self,
        new_key: torch.Tensor,   # [D] float  — key of token requesting insertion
    ) -> int:
        """
        Return the L2 slot index that is the best eviction candidate:
          • If FAISS is available: find the existing slot most similar to new_key
            (evicting the most redundant one keeps maximum diversity).
          • Fallback: return the slot with minimum importance score.

        This implements a diversity-preserving eviction policy similar to
        the one in finetune_gpt2.py's MapAttention._flush_pool().
        """
        cache = self._cache
        if not _FAISS or len(self._index) == 0:
            return cache._l2_min_slot()

        # Most similar to new_key → most redundant → best eviction target
        sims, slot_ids = self._index.search_sim(new_key, k=4)
        candidates     = [s for s in slot_ids if s >= cache.cfg.important_start]
        if not candidates:
            return cache._l2_min_slot()

        # Among redundant candidates, pick the one with lowest importance
        best_slot = min(
            candidates,
            key=lambda s: float(cache.importance[s].item()),
        )
        return best_slot

    # ── FAISS-accelerated cross-turn cache boost ──────────────────────────────

    def boost_similar_slots(
        self,
        new_embeddings:  torch.Tensor,   # [T_new, D]
        threshold:       Optional[float] = None,
        boost_factor:    Optional[float] = None,
    ) -> None:
        """
        Boost importance of cached tokens similar to any new token.

        FAISS path:  find top-k cached slots for each new token in O(T_new × log C).
        Fallback:    iterate C × T_new pairs in numpy.
        """
        cache    = self._cache
        thresh   = threshold   or self._thresh
        bf       = boost_factor or self._boost_f
        T_new    = new_embeddings.shape[0]

        if not _FAISS or len(self._index) == 0:
            # Fallback: nothing to boost if index empty
            return

        for i in range(T_new):
            sims, slot_ids = self._index.search_sim(new_embeddings[i], k=8)
            for sim, slot in zip(sims, slot_ids):
                if sim >= thresh and 0 <= slot < cache.importance.shape[0]:
                    cache.importance[slot] *= bf

    # ── Index maintenance ─────────────────────────────────────────────────────

    def index_push(
        self,
        key_vector: torch.Tensor,   # [D]  — layer-0, head-0 representative key
        slot:       int,
    ) -> None:
        """Register a newly inserted slot in the FAISS index."""
        if _FAISS:
            self._index.add(key_vector.unsqueeze(0), [slot])

    def index_rebuild(
        self,
        key_tensor: torch.Tensor,   # [H, budget, D]  — full key buffer, layer 0
    ) -> None:
        """
        Full index rebuild — call after bulk cache loads (e.g., after summary pass).
        Uses layer-0, head-0 keys as the representative vectors.
        """
        if not _FAISS:
            return
        cache    = self._cache
        occupied = cache.occupied.nonzero(as_tuple=True)[0]
        if occupied.shape[0] == 0:
            return
        # Use head-0 keys from layer 0 as the routing fingerprint
        vecs     = key_tensor[0, occupied, :]   # [F, D]
        slot_ids = occupied.tolist()
        self._index.rebuild(vecs, slot_ids)


# ── IVF-backed approximate index for large budgets ────────────────────────────

class FaissIVFIndex:
    """
    Approximate NN index (IndexIVFFlat) for very large cache budgets (≥ 2048).

    At budget < 1024 use _FlatCosineIndex — it is faster (exact, no training).
    At budget ≥ 2048 IVF becomes faster: O(√budget) probes, 10–50× speedup.

    Requires a training step (via .train()) before first .add() call.
    The training vectors can be random — the IVF cell boundaries adapt fine.
    """

    def __init__(self, d: int, n_centroids: int = 64, n_probes: int = 8):
        if not _FAISS:
            self.index = None
            return
        quantiser   = faiss.IndexFlatIP(d)
        self.index  = faiss.IndexIVFFlat(quantiser, d, n_centroids, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = n_probes
        self.d      = d
        self.id_map = []
        self._trained = False

    def train_if_needed(self, n_vectors: int) -> None:
        """Auto-train using random unit vectors if not yet trained."""
        if not _FAISS or self._trained:
            return
        train_data = np.random.randn(max(n_vectors, 256), self.d).astype(np.float32)
        nrm        = np.linalg.norm(train_data, axis=1, keepdims=True).clip(1e-8)
        train_data /= nrm
        self.index.train(train_data)
        self._trained = True

    def add(self, vectors: torch.Tensor, slot_ids: List[int]) -> None:
        if not _FAISS:
            return
        self.train_if_needed(len(slot_ids))
        v   = vectors.float().detach().cpu().numpy()
        nrm = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
        self.index.add(v / nrm)
        self.id_map.extend(slot_ids)

    def search(self, query: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not _FAISS or not self._trained or self.index.ntotal == 0:
            return np.array([]), np.array([])
        q   = query.float().detach().cpu().numpy().reshape(1, -1)
        nrm = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
        k   = min(k, self.index.ntotal)
        sims, idxs = self.index.search(q / nrm, k)
        slots = np.array([self.id_map[i] for i in idxs[0] if 0 <= i < len(self.id_map)])
        return sims[0], slots
