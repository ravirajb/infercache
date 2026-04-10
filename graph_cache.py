"""
inference/graph_cache.py
--------------------------
Disconnected Graph KV Cache — depth-adaptive cluster-based retrieval.

The idea
--------
The L2 "important" pool in HierarchicalKVCache stores tokens in a flat array.
Finding the right tokens for a new query requires scoring all F² slots against
the query key — O(F) per head per decode step.

Instead, organise the important pool into a small set of disconnected semantic
subgraphs (clusters).  At decode time:
  1. Score the query key against cluster CENTROIDS only:  O(n_clusters)
  2. Retrieve tokens from the top-1 (or top-2) relevant cluster: O(cluster_size)

Total cost: O(n_clusters + cluster_size) << O(budget)

This mirrors how human memory works: a topic-specific "context cluster"
is activated, not the entire memory store.

Cluster formation
-----------------
  Online k-means over key embeddings:  O(n_clusters × D) per new token.
  On token eviction, the cluster is updated (centroid -= evicted, re-normalised).
  No batch recomputation needed — fully online.

  We use cosine similarity (not Euclidean) because key vectors in transformers
  have near-unit norm after RoPE rotation — cosine similarity ≈ dot product.

Varying context / depth adaptation
-----------------------------------
  Short prompt (T ≤ 512):   1–2 clusters active, rest empty.
  Medium prompt (T ~ 2K):   3–4 clusters populated.
  Deep reasoning chain:     All n_clusters filled, each encodes a different
                            reasoning branch.

  The engine selects clusters whose centroid dot-product with the query
  exceeds a threshold, not a fixed top-k — so it adapts to prompt depth:
    shallow prompt → 1 cluster retrieved (< budget/n_clusters keys attended)
    deep prompt    → all clusters retrieved (≈ full budget keys attended)

Connection to "multiple hierarchical / graph-based caches"
----------------------------------------------------------
  Each cluster IS a mini cache — its own K/V slice, centroid, and fill level.
  MultiClusterCache below gives each domain (CODE/QA/LANGUAGE) its own set
  of clusters, for a total of domain_count × n_clusters independent mini-caches.

Memory cost
-----------
  Each cluster: capacity × num_kv_heads × head_dim × bytes
  n_clusters=16, capacity=32, H=8, D=128, fp16:
    16 × 32 × 8 × 128 × 2 = 8 MB per layer
    × 28 layers = 224 MB — modest vs 4.7 GB standard KV at 8K context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["ClusterConfig", "GraphKVCache", "MultiClusterCache"]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ClusterConfig:
    num_layers:   int
    num_kv_heads: int
    head_dim:     int

    n_clusters:       int   = 16    # number of disconnected subgraphs
    cluster_capacity: int   = 32    # max tokens per cluster
    sim_threshold:    float = 0.3   # cos-sim to centroid required for cluster activation
    sink_size:        int   = 4     # L0 sinks (always retrieved, cluster-independent)
    recent_size:      int   = 128   # L1 recent window (still present, cluster-independent)

    dtype: torch.dtype = torch.float16

    @property
    def cluster_budget(self) -> int:
        return self.n_clusters * self.cluster_capacity

    @property
    def total_budget(self) -> int:
        return self.sink_size + self.recent_size + self.cluster_budget


# ── Single cluster ────────────────────────────────────────────────────────────

class _Cluster:
    """
    One semantic subgraph/cluster.  Stores:
      - K/V tensors for tokens assigned to this cluster
      - A running centroid = mean of all key vectors (first layer, first head)
        used as a cheap proxy for cluster content similarity
      - Fill level and position tracking
    """

    def __init__(
        self,
        cluster_id:   int,
        num_layers:   int,
        num_kv_heads: int,
        head_dim:     int,
        capacity:     int,
        dtype:        torch.dtype,
        device:       torch.device,
    ):
        self.id       = cluster_id
        self.capacity = capacity
        self.dtype    = dtype

        shape = (num_layers, num_kv_heads, capacity, head_dim)
        self.K         = torch.zeros(shape, dtype=dtype, device=device)
        self.V         = torch.zeros(shape, dtype=dtype, device=device)
        self.positions = torch.full((capacity,), -1, dtype=torch.long)
        self.scores    = torch.zeros(capacity, dtype=torch.float32)   # importance
        self.fill      = 0

        # Centroid: mean key from layer-0, head-0
        self.centroid  = torch.zeros(head_dim, dtype=torch.float32, device=device)
        self._c_count  = 0

    def is_full(self) -> bool:
        return self.fill >= self.capacity

    def is_empty(self) -> bool:
        return self.fill == 0

    def cos_sim_to_query(self, q_key: torch.Tensor) -> float:
        """Cosine similarity between this cluster's centroid and a query key vector."""
        if self._c_count == 0:
            return -1.0
        q  = q_key.float().squeeze()
        c  = self.centroid
        qs = q.norm(dim=-1).clamp(min=1e-6)
        cs = c.norm(dim=-1).clamp(min=1e-6)
        return float((q / qs * c / cs).sum().item())

    def insert(
        self,
        K_token: torch.Tensor,   # [num_layers, num_kv_heads, 1, head_dim]
        V_token: torch.Tensor,
        position: int,
        score:    float,
    ) -> Optional[int]:
        """
        Insert a token.  If full, evict the slot with minimum importance  score.
        Returns the evicted slot index, or None if no eviction was needed.
        """
        evicted = None
        if self.is_full():
            # Evict minimum importance
            evicted = int(self.scores[:self.fill].argmin().item())
            # Remove evicted from centroid (approximate — subtract its contribution)
            old_key = self.K[0, 0, evicted, :].float()
            n       = float(self._c_count)
            if n > 1:
                self.centroid = (self.centroid * n - old_key) / (n - 1)
                self._c_count -= 1
            else:
                self.centroid.zero_()
                self._c_count = 0
            slot = evicted
        else:
            slot = self.fill
            self.fill += 1

        self.K[:, :, slot, :] = K_token[:, :, 0, :].to(self.K.dtype)
        self.V[:, :, slot, :] = V_token[:, :, 0, :].to(self.V.dtype)
        self.positions[slot]  = position
        self.scores[slot]     = score

        # Update centroid
        new_key = self.K[0, 0, slot, :].float()
        n       = float(self._c_count)
        self.centroid = (self.centroid * n + new_key) / (n + 1)
        self._c_count += 1

        return evicted

    def to_kv_tensors(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Export occupied K/V as [1, H, fill, D] — HF format.
        Returns K, V sorted by position.
        """
        if self.fill == 0:
            return None, None
        pos   = self.positions[:self.fill]
        order = pos.argsort()
        K_out = self.K[:, :, order, :]   # [L, H, fill, D]
        V_out = self.V[:, :, order, :]
        if device is not None:
            K_out = K_out.to(device)
            V_out = V_out.to(device)
        return K_out, V_out

    def reset(self) -> None:
        self.K.zero_()
        self.V.zero_()
        self.positions.fill_(-1)
        self.scores.zero_()
        self.fill      = 0
        self.centroid.zero_()
        self._c_count  = 0


# ── Graph KV Cache ────────────────────────────────────────────────────────────

class GraphKVCache:
    """
    Disconnected cluster-based KV cache.

    Three regions:
      L0 — attention sinks:        never evicted, always in output (small, 4–8 tokens)
      L1 — recent window:          FIFO ring, always in output
      L2 — cluster graph:          n_clusters × cluster_capacity  tokens
                                   cluster selected at decode time by centroid similarity

    Decode-time retrieval is O(n_clusters × D) for centroid similarity scoring,
    then O(cluster_size) to fetch the relevant cluster's K/V.
    Compare: flat important pool costs O(budget).

    For a deeply reasoning context (many clusters relevant), retrieval degrades
    gracefully: all clusters are retrieved → equivalent to the flat pool.
    """

    def __init__(self, cfg: ClusterConfig, device=None):
        self.cfg    = cfg
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        D = cfg.head_dim
        H = cfg.num_kv_heads
        L = cfg.num_layers

        # ── L0 sink ──────────────────────────────────────────────────────
        sink_shape = (L, H, cfg.sink_size, D)
        self._sink_K    = torch.zeros(sink_shape, dtype=cfg.dtype, device=self.device)
        self._sink_V    = torch.zeros(sink_shape, dtype=cfg.dtype, device=self.device)
        self._sink_pos  = torch.full((cfg.sink_size,), -1, dtype=torch.long)
        self._sink_fill = 0

        # ── L1 recent ring ───────────────────────────────────────────────
        recent_shape = (L, H, cfg.recent_size, D)
        self._rec_K    = torch.zeros(recent_shape, dtype=cfg.dtype, device=self.device)
        self._rec_V    = torch.zeros(recent_shape, dtype=cfg.dtype, device=self.device)
        self._rec_pos  = torch.full((cfg.recent_size,), -1, dtype=torch.long)
        self._rec_ptr  = 0
        self._rec_fill = 0

        # ── L2 cluster graph ─────────────────────────────────────────────
        self._clusters: List[_Cluster] = [
            _Cluster(
                cluster_id   = i,
                num_layers   = L,
                num_kv_heads = H,
                head_dim     = D,
                capacity     = cfg.cluster_capacity,
                dtype        = cfg.dtype,
                device       = self.device,
            )
            for i in range(cfg.n_clusters)
        ]
        self._next_cluster  = 0   # round-robin assignment for new tokens

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_clusters(self) -> int:
        return self.cfg.n_clusters

    @property
    def active_cluster_count(self) -> int:
        return sum(1 for c in self._clusters if not c.is_empty())

    # ── Token insertion ───────────────────────────────────────────────────────

    def push(
        self,
        key_states:   torch.Tensor,   # [L, H, 1, D]
        value_states: torch.Tensor,
        position:     int,
        score:        float = 0.0,    # importance score from attention mass
    ) -> None:
        """
        Insert a token into L1 (recent), displacing to cluster graph if needed.
        Cluster assignment is based on cosine similarity to existing centroids.
        """
        # ── L0 sink (fill first 4–8 absolute positions) ─────────────────
        if position < self.cfg.sink_size and self._sink_fill < self.cfg.sink_size:
            s = self._sink_fill
            self._sink_K[:, :, s, :] = key_states[:, :, 0, :]
            self._sink_V[:, :, s, :] = value_states[:, :, 0, :]
            self._sink_pos[s]         = position
            self._sink_fill          += 1
            return

        # ── L1 recent ring ────────────────────────────────────────────────
        # Displace current slot into cluster graph before overwriting
        ptr = self._rec_ptr
        if self._rec_pos[ptr] >= 0:
            # Displaced token's key (layer-0, head-0) for cluster routing
            old_k = self._rec_K[:, :, ptr:ptr+1, :]   # [L, H, 1, D]
            old_v = self._rec_V[:, :, ptr:ptr+1, :]
            old_pos   = int(self._rec_pos[ptr].item())
            old_score = score  # use current importance proxy
            self._insert_to_cluster(old_k, old_v, old_pos, old_score)

        # Write new token into the ring slot
        self._rec_K[:, :, ptr, :] = key_states[:, :, 0, :]
        self._rec_V[:, :, ptr, :] = value_states[:, :, 0, :]
        self._rec_pos[ptr]         = position
        self._rec_fill = min(self._rec_fill + 1, self.cfg.recent_size)
        self._rec_ptr  = (ptr + 1) % self.cfg.recent_size

    def _insert_to_cluster(
        self,
        K_token: torch.Tensor,  # [L, H, 1, D]
        V_token: torch.Tensor,
        position: int,
        score:    float,
    ) -> None:
        """
        Assign token to the cluster whose centroid it is most similar to.
        If all clusters are empty, assign round-robin.
        If similarity is below all thresholds, assign to the least-filled cluster.
        """
        q_key    = K_token[0, 0, 0, :]   # [D] — use layer-0, head-0 as routing key
        best_i   = -1
        best_sim = -2.0

        for i, cluster in enumerate(self._clusters):
            sim = cluster.cos_sim_to_query(q_key)
            if sim > best_sim:
                best_sim = sim
                best_i   = i

        # Use best-matching cluster if sim > threshold, else least-filled empty one
        if best_sim < self.cfg.sim_threshold:
            # Try to find an empty/sparse cluster to start a new subgraph
            for i, c in enumerate(self._clusters):
                if c.is_empty():
                    best_i = i
                    break
            else:
                # All clusters occupied — use least-filled
                fills = [c.fill for c in self._clusters]
                best_i = int(min(range(self.n_clusters), key=lambda i: fills[i]))

        self._clusters[best_i].insert(K_token, V_token, position, score)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_key:     torch.Tensor,              # [H, D]  query key for current token
        threshold:     Optional[float]  = None,   # override cluster selection threshold
        top_n_clusters: Optional[int]   = None,   # override: always retrieve top-n
        device:        Optional[torch.device] = None,
    ) -> Tuple[Optional[object], torch.Tensor]:
        """
        Build HF past_key_values from:
          L0 sinks  +  recent window  +  relevant clusters (by centroid similarity)

        Returns:
          past_kv : tuple of (K, V) per layer in HF format, or None if empty
          selected_clusters : bool tensor [n_clusters] — which clusters were used
        """
        dev   = device or self.device
        thresh = threshold if threshold is not None else self.cfg.sim_threshold
        L, H, _, D = self._sink_K.shape

        # ── Determine which clusters to retrieve ──────────────────────────
        q_key_mean = query_key.float().mean(dim=0)   # [D] — average over heads
        sims       = torch.tensor([
            c.cos_sim_to_query(q_key_mean) for c in self._clusters
        ])
        selected = torch.zeros(self.n_clusters, dtype=torch.bool)

        if top_n_clusters is not None:
            top_k_idx = sims.topk(min(top_n_clusters, self.n_clusters)).indices
            selected[top_k_idx] = True
        else:
            selected = sims >= thresh
            if not selected.any():
                # Always retrieve at least the best cluster
                selected[sims.argmax()] = True

        # ── Assemble per-layer K/V ─────────────────────────────────────────
        past_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for li in range(L):
            parts_k: List[torch.Tensor] = []
            parts_v: List[torch.Tensor] = []

            # Sinks
            if self._sink_fill > 0:
                parts_k.append(self._sink_K[li, :, :self._sink_fill, :])
                parts_v.append(self._sink_V[li, :, :self._sink_fill, :])

            # Recent
            if self._rec_fill > 0:
                # Get recent tokens in position order
                rec_pos = self._rec_pos[:self._rec_fill]
                order   = rec_pos.argsort()
                parts_k.append(self._rec_K[li, :, order, :])
                parts_v.append(self._rec_V[li, :, order, :])

            # Relevant clusters
            for i, cluster in enumerate(self._clusters):
                if not selected[i] or cluster.is_empty():
                    continue
                K_c, V_c = cluster.to_kv_tensors(device=None)
                if K_c is not None:
                    parts_k.append(K_c[li])   # [H, fill, D]
                    parts_v.append(V_c[li])

            if not parts_k:
                past_kv.append(None)
                continue

            K_li = torch.cat(parts_k, dim=1).unsqueeze(0).to(dev)   # [1, H, F, D]
            V_li = torch.cat(parts_v, dim=1).unsqueeze(0).to(dev)
            past_kv.append((K_li, V_li))

        if all(x is None for x in past_kv):
            return None, selected

        return tuple(past_kv), selected

    # ── Bulk load from summary pass ───────────────────────────────────────────

    def load_from_summary(
        self,
        past_key_values,                    # HF past_kv
        importance_scores: torch.Tensor,    # [T] float
        positions:         Optional[torch.Tensor] = None,
    ) -> None:
        """Populate the graph cache from a summary-pass KV snapshot."""
        self.reset()
        from .hierarchical_cache import _normalise_past_kv
        pkv     = _normalise_past_kv(past_key_values)
        seq_len = pkv[0][0].shape[2]
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.long)

        L = len(pkv)
        for tok in range(seq_len):
            pos   = int(positions[tok].item())
            score = float(importance_scores[tok].item())

            # Build [L, H, 1, D] for this token
            K_tok = torch.stack([pkv[li][0][0, :, tok, :] for li in range(L)], 0).unsqueeze(2)
            V_tok = torch.stack([pkv[li][1][0, :, tok, :] for li in range(L)], 0).unsqueeze(2)

            self.push(K_tok, V_tok, pos, score)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._sink_K.zero_(); self._sink_V.zero_()
        self._sink_pos.fill_(-1); self._sink_fill = 0

        self._rec_K.zero_(); self._rec_V.zero_()
        self._rec_pos.fill_(-1); self._rec_ptr = 0; self._rec_fill = 0

        for c in self._clusters:
            c.reset()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def cluster_summary(self) -> List[dict]:
        """Return per-cluster fill, centroid norm, and mean importance."""
        return [
            {
                "id":       c.id,
                "fill":     c.fill,
                "capacity": c.capacity,
                "centroid_norm": float(c.centroid.norm().item()),
                "mean_score":    float(c.scores[:c.fill].mean().item()) if c.fill > 0 else 0.0,
            }
            for c in self._clusters
        ]


# ── Multi-domain graph cache ──────────────────────────────────────────────────

class MultiClusterCache:
    """
    One GraphKVCache per domain (CODE / QA / LANGUAGE).

    Each domain maintains completely independent cluster graphs —
    code tokens never compete with Q&A tokens for cluster slots.

    Usage::
        mcc = MultiClusterCache(cluster_cfg, device='cuda')
        mcc.activate_domain('code')
        mcc.active_cache.push(...)
        past_kv, which_clusters = mcc.active_cache.retrieve(q_key)
    """

    _DOMAINS = ("code", "qa", "language")

    def __init__(self, cfg: ClusterConfig, device=None):
        self._caches = {
            d: GraphKVCache(cfg, device=device)
            for d in self._DOMAINS
        }
        self._active = "language"

    def activate_domain(self, domain: str) -> None:
        if domain not in self._DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from {self._DOMAINS}")
        self._active = domain

    @property
    def active_cache(self) -> GraphKVCache:
        return self._caches[self._active]

    def get(self, domain: str) -> GraphKVCache:
        return self._caches[domain]

    def reset_all(self) -> None:
        for c in self._caches.values():
            c.reset()

    def summary(self) -> dict:
        return {
            domain: cache.cluster_summary()
            for domain, cache in self._caches.items()
        }
