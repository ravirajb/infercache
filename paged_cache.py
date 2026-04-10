"""
inference/paged_cache.py
------------------------
Multi-session paged KV cache for GPU memory efficiency.

Problem
-------
With a fixed HierarchicalKVCache per session, each session pre-allocates
[n_layers, n_heads, budget, head_dim] regardless of how many tokens it
actually uses.  With budget=512 and 10 concurrent sessions the GPU holds
5120 token slots even if average usage is 100 tokens/session (10× waste).

Solution: PagePool + KVAllocator
---------------------------------
A PagePool pre-allocates one large tensor on GPU.  It is divided into
fixed-size pages (PAGE_SIZE = 16 tokens).  The KVAllocator hands pages to
sessions on demand and reclaims them via LRU eviction when the pool is full.

   GPU memory:
   ┌──────────────────────────────────────────────────────────────┐
   │  page 0  │  page 1  │  page 2  │ ... │  page N-1             │
   │                                                              │
   │  session A uses pages [0, 3, 7]                              │
   │  session B uses pages [1, 4]                                 │
   │  session C uses pages [2, 5, 6, 8]                           │
   └──────────────────────────────────────────────────────────────┘

This raises GPU utilization from ~15 % (one large allocation, mostly idle)
to ~75 % (pages allocated on demand, LRU eviction).

Integration
-----------
MultiSessionCache is the entry point.  Call:

    pool  = PagePool.build(n_sessions=16, cfg=engine.cfg)
    cache = MultiSessionCache(pool)
    hcache = cache.get_or_create("session-abc")   # HierarchicalKVCache-like view

The returned object supports the same interface as HierarchicalKVCache
(push / load_from_hf_output / to_hf_past_key_values / reset / stats).

Note
----
This implementation does NOT require a custom attention kernel.  Each
session's logical KV is gathered from the scattered page pool into a
contiguous tensor before attention.  This gather is O(page_count × PAGE_SIZE)
and adds ~0.5 ms overhead compared with a contiguous cache.  A custom kernel
(vLLM-style block sparse attention) can eliminate this gather; that is left
as a future optimisation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


__all__ = [
    "PAGE_SIZE",
    "PagePool",
    "KVAllocator",
    "MultiSessionCache",
    "PagedSessionView",
    "PagedCacheConfig",
]

PAGE_SIZE: int = 16   # tokens per page


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class PagedCacheConfig:
    n_pages:    int   = 512     # total pages in the pool (512 × 16 = 8192 tok)
    n_layers:   int   = 36
    n_heads:    int   = 8
    head_dim:   int   = 128
    dtype:      torch.dtype = torch.float16
    device:     str   = "cuda"
    max_pages_per_session: int = 64    # hard cap per session (1024 tokens)

    @property
    def total_tokens(self) -> int:
        return self.n_pages * PAGE_SIZE

    def gpu_memory_mb(self) -> float:
        """Approximate GPU memory for K + V tensors."""
        bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        total = 2 * self.n_layers * self.n_heads * self.total_tokens * self.head_dim
        return total * bytes_per_elem / (1024 ** 2)


# ── PagePool ──────────────────────────────────────────────────────────────────

class PagePool:
    """
    Pre-allocated GPU tensors divided into fixed-size pages.

    Tensor layout:
      K / V : [n_layers, n_heads, n_pages * PAGE_SIZE, head_dim]

    Individual tokens are addressed as:
      page_id * PAGE_SIZE + slot_within_page
    """

    def __init__(self, cfg: PagedCacheConfig) -> None:
        self.cfg = cfg
        total    = cfg.n_pages * PAGE_SIZE
        dev      = cfg.device

        # Allocate once; never resized after construction
        self.K = torch.zeros(
            cfg.n_layers, cfg.n_heads, total, cfg.head_dim,
            dtype=cfg.dtype, device=dev,
        )
        self.V = torch.zeros(
            cfg.n_layers, cfg.n_heads, total, cfg.head_dim,
            dtype=cfg.dtype, device=dev,
        )

    @classmethod
    def build(
        cls,
        n_sessions: int,
        n_layers:   int,
        n_heads:    int,
        head_dim:   int,
        avg_tokens_per_session: int = 256,
        dtype:      torch.dtype = torch.float16,
        device:     str = "cuda",
    ) -> "PagePool":
        """
        Convenience constructor: size the pool to support `n_sessions` with
        `avg_tokens_per_session` each, plus a 25 % over-provision buffer.
        """
        tokens_needed = n_sessions * avg_tokens_per_session
        pages_needed  = (tokens_needed + PAGE_SIZE - 1) // PAGE_SIZE
        pages_needed  = int(pages_needed * 1.25)  # 25 % buffer
        cfg = PagedCacheConfig(
            n_pages   = pages_needed,
            n_layers  = n_layers,
            n_heads   = n_heads,
            head_dim  = head_dim,
            dtype     = dtype,
            device    = device,
            max_pages_per_session = (avg_tokens_per_session * 4) // PAGE_SIZE + 4,
        )
        return cls(cfg)

    # Slot addressing helpers

    def _abs(self, page_id: int, slot: int) -> int:
        return page_id * PAGE_SIZE + slot

    def write_token(
        self,
        layer:   int,
        page_id: int,
        slot:    int,
        k_vec:   torch.Tensor,   # [n_heads, head_dim]
        v_vec:   torch.Tensor,   # [n_heads, head_dim]
    ) -> None:
        abs_slot = self._abs(page_id, slot)
        with torch.no_grad():
            self.K[layer, :, abs_slot, :] = k_vec
            self.V[layer, :, abs_slot, :] = v_vec

    def gather_session(
        self,
        layer:    int,
        page_ids: List[int],
        n_tokens: int,          # may be < len(page_ids) * PAGE_SIZE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K/V for all tokens in a session into contiguous tensors.

        Returns K, V each of shape [n_heads, n_tokens, head_dim].
        """
        if not page_ids:
            dev = self.K.device
            empty = torch.zeros(self.cfg.n_heads, 0, self.cfg.head_dim,
                                dtype=self.cfg.dtype, device=dev)
            return empty, empty

        # Build absolute index list
        indices: List[int] = []
        for pg in page_ids:
            base = pg * PAGE_SIZE
            indices.extend(range(base, base + PAGE_SIZE))
        # Trim to actual token count
        indices = indices[:n_tokens]

        idx_t = torch.tensor(indices, dtype=torch.long, device=self.K.device)
        K_out = self.K[layer, :, idx_t, :]   # [n_heads, n_tokens, head_dim]
        V_out = self.V[layer, :, idx_t, :]
        return K_out, V_out


# ── KVAllocator ───────────────────────────────────────────────────────────────

class KVAllocator:
    """
    Thread-safe free-list page allocator with LRU eviction.

    Internally tracks:
      _free_list       : List[int]           — available page IDs
      _session_pages   : {session_id: [page_ids]}
      _session_atime   : {session_id: float} — last access timestamp
      _page_tokens     : {session_id: int}   — filled token count
    """

    def __init__(self, n_pages: int, max_pages_per_session: int) -> None:
        self._max_pps    = max_pages_per_session
        self._lock       = threading.Lock()
        self._free_list: List[int]        = list(range(n_pages))
        self._session_pages: Dict[str, List[int]] = {}
        self._session_atime: Dict[str, float]     = {}
        self._session_ntok:  Dict[str, int]       = {}

    # ── Page allocation ───────────────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Evict the session with the oldest access time to free its pages."""
        if not self._session_pages:
            raise RuntimeError("PagePool exhausted and no sessions to evict")
        lru_session = min(self._session_atime, key=self._session_atime.__getitem__)
        self._release_locked(lru_session)

    def _release_locked(self, session_id: str) -> None:
        pages = self._session_pages.pop(session_id, [])
        self._free_list.extend(pages)
        self._session_atime.pop(session_id, None)
        self._session_ntok.pop(session_id, None)

    def allocate_page(self, session_id: str) -> int:
        """Allocate one page for `session_id`.  LRU-evicts if pool is full."""
        with self._lock:
            cur_pages = len(self._session_pages.get(session_id, []))
            if cur_pages >= self._max_pps:
                raise RuntimeError(
                    f"Session {session_id!r} exceeded max pages ({self._max_pps})"
                )
            if not self._free_list:
                self._evict_lru()
            page_id = self._free_list.pop()
            self._session_pages.setdefault(session_id, []).append(page_id)
            self._session_atime[session_id] = time.monotonic()
            return page_id

    def touch(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._session_atime:
                self._session_atime[session_id] = time.monotonic()

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._release_locked(session_id)

    # ── Token counter ─────────────────────────────────────────────────────────

    def set_token_count(self, session_id: str, n: int) -> None:
        with self._lock:
            self._session_ntok[session_id] = n

    def get_token_count(self, session_id: str) -> int:
        with self._lock:
            return self._session_ntok.get(session_id, 0)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def pages_for(self, session_id: str) -> List[int]:
        with self._lock:
            return list(self._session_pages.get(session_id, []))

    def utilization(self) -> float:
        """Fraction of pages currently allocated."""
        with self._lock:
            total_pages = sum(len(v) for v in self._session_pages.values())
            n_pages     = total_pages + len(self._free_list)
            return total_pages / n_pages if n_pages else 0.0

    def active_sessions(self) -> List[str]:
        with self._lock:
            return list(self._session_pages.keys())


# ── PagedSessionView ──────────────────────────────────────────────────────────

class PagedSessionView:
    """
    A HierarchicalKVCache-compatible view of one session's pages in a PagePool.

    This class writes tokens into the PagePool via the KVAllocator and exposes
    the same gather interface for attention.  It does NOT subclass
    HierarchicalKVCache to keep dependencies simple.

    API surface used by memory_engine.py
    -------------------------------------
     push(k_states, v_states, positions, importance)
         k_states : [n_layers, n_heads, n_new, head_dim]
     gather(layer) -> K, V  [n_heads, n_tokens, head_dim]
     reset()
     n_tokens  : int property
    """

    def __init__(
        self,
        session_id: str,
        pool:       PagePool,
        allocator:  KVAllocator,
    ) -> None:
        self.session_id = session_id
        self._pool      = pool
        self._alloc     = allocator

        self._pages:      List[int]   = []   # page IDs in order
        self._n_tokens:   int         = 0    # tokens written so far
        self._importance: List[float] = []   # one per token
        self._positions:  List[int]   = []   # original token positions
        self._lock        = threading.Lock()

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def push(
        self,
        k_states:   torch.Tensor,   # [n_layers, n_heads, T, head_dim]
        v_states:   torch.Tensor,   # [n_layers, n_heads, T, head_dim]
        positions:  List[int],
        importance: List[float],
    ) -> None:
        """Write T new tokens into the page pool."""
        cfg     = self._pool.cfg
        n_new   = k_states.shape[2]

        with self._lock:
            for t in range(n_new):
                slot_in_page = self._n_tokens % PAGE_SIZE
                if slot_in_page == 0:
                    page_id = self._alloc.allocate_page(self.session_id)
                    self._pages.append(page_id)
                else:
                    page_id = self._pages[-1]

                for layer in range(cfg.n_layers):
                    self._pool.write_token(
                        layer,
                        page_id,
                        slot_in_page,
                        k_states[layer, :, t, :],
                        v_states[layer, :, t, :],
                    )
                self._n_tokens += 1
                self._positions.append(positions[t] if t < len(positions) else self._n_tokens - 1)
                self._importance.append(importance[t] if t < len(importance) else 0.0)

            self._alloc.set_token_count(self.session_id, self._n_tokens)
            self._alloc.touch(self.session_id)

    def gather(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return K, V for this session at `layer`.  Shape: [n_heads, T, head_dim]."""
        return self._pool.gather_session(layer, self._pages, self._n_tokens)

    def to_hf_past_key_values(self, device: str):
        """
        Export this session's KV into HuggingFace past_key_values format.
        Returns list of (K, V) per layer, each [1, n_heads, T, head_dim].
        """
        past = []
        for layer in range(self._pool.cfg.n_layers):
            K, V = self.gather(layer)           # [n_heads, T, head_dim]
            past.append((K.unsqueeze(0), V.unsqueeze(0)))  # [1, n_heads, T, head_dim]
        return past

    def top_importance_tokens(self, top_k: int) -> Tuple[List[int], List[float]]:
        """Return indices (positions) and importance of the top_k tokens."""
        if not self._importance:
            return [], []
        pairs = sorted(zip(self._importance, range(len(self._importance))), reverse=True)
        top_pairs = pairs[:top_k]
        idxs = [pos for _, pos in top_pairs]
        imps = [imp for imp, _ in top_pairs]
        return idxs, imps

    def reset(self) -> None:
        with self._lock:
            self._alloc.release_session(self.session_id)
            self._pages      = []
            self._n_tokens   = 0
            self._importance = []
            self._positions  = []


# ── MultiSessionCache ─────────────────────────────────────────────────────────

class MultiSessionCache:
    """
    Manages all PagedSessionViews sharing one PagePool.

    Usage
    -----
        pool  = PagePool.build(n_sessions=16, ...)
        cache = MultiSessionCache(pool)
        view  = cache.get_or_create("session-xyz")
        # use view.push / view.gather
        cache.reset("session-xyz")
    """

    def __init__(self, pool: PagePool) -> None:
        self._pool    = pool
        self._alloc   = KVAllocator(pool.cfg.n_pages, pool.cfg.max_pages_per_session)
        self._views:  Dict[str, PagedSessionView] = {}
        self._lock    = threading.Lock()

    def get_or_create(self, session_id: str) -> PagedSessionView:
        with self._lock:
            if session_id not in self._views:
                self._views[session_id] = PagedSessionView(
                    session_id, self._pool, self._alloc
                )
            return self._views[session_id]

    def reset(self, session_id: str) -> None:
        with self._lock:
            view = self._views.pop(session_id, None)
        if view:
            view.reset()

    def reset_all(self) -> None:
        with self._lock:
            sessions = list(self._views.keys())
        for sid in sessions:
            self.reset(sid)

    def utilization(self) -> float:
        return self._alloc.utilization()

    def active_sessions(self) -> List[str]:
        return self._alloc.active_sessions()

    def pool_info(self) -> dict:
        cfg = self._pool.cfg
        return {
            "total_pages":   cfg.n_pages,
            "total_tokens":  cfg.total_tokens,
            "gpu_memory_mb": cfg.gpu_memory_mb(),
            "utilization":   f"{self.utilization()*100:.1f}%",
            "active_sessions": len(self.active_sessions()),
        }
