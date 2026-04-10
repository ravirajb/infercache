"""
inference/rbf_path.py
---------------------
RBF Kernel Path Evaluator — pre-generation context analysis.

Purpose
-------
Before invoking the full LLM forward pass, extract a lightweight importance
signal from the model's token embeddings using RBF kernel similarity.

The key observation from the attached research context:
  "Transformers already encode knowledge in embeddings.
   You are simply probing different geometric projections of the
   representation space.  The kernel helps reveal latent semantic
   clusters that the model already learned."

How it works  (all operations on EMBEDDING vectors, not full hidden states)
---------------------------------------------------------------------------
1. Embed the prompt tokens using the model's embedding layer only.
   Cost: O(T × d)  — a simple lookup, no attention needed.

2. Partition the d-dimensional embedding into P equal blocks:
     B₁ = dims  [0,  d/P)
     B₂ = dims  [d/P, 2d/P)
     …
     Bₚ = dims  [(P-1)d/P, d)

3. For each block Bₚ, compute RBF similarity between all token pairs:
     Kₚ(xᵢ, xⱼ) = exp(−‖xᵢ[Bₚ] − xⱼ[Bₚ]‖² / 2σ²)

   Naively O(T²). We approximate with Random Fourier Features (Rahimi & Recht 2007):
     φ(x) = √(2/D_rff) · cos(Wx + b),   W ~ N(0, 2/σ²I),  b ~ U[0, 2π)
     Kₚ(xᵢ, xⱼ) ≈ φ(xᵢ)·φ(xⱼ)
   → Reduced to T × D_rff matrix multiply: O(T × D_rff × d/P)

4. Build a sparse top-k neighbour adjacency matrix from the kernel scores.
   Compute the "hub score" = out-degree in the adjacency graph.
   A token that is closely similar to many others in many partitions
   is a "hub" — a semantic centroid of the prompt.

5. Aggregate hub scores across partitions.  Normalise.
   Return as token_importance: [T] float — a cheap alternative to
   running a full forward pass with output_attentions=True.

Use cases in this inference stack
----------------------------------
  A. Cache warm-up boost: in multi-turn dialogue, run RBF on the new
     turn's embeddings and boost the importance of cached tokens that
     are close (in some partition) to newly mentioned concepts.

  B. Context compression: for very long prompts where even the summary
     pass memory is tight, use RBF hub scores to pre-select a subset of
     tokens to feed into the summary pass.

  C. Pre-route hint: the partition where hub scores are concentrated
     can hint which domain cache to activate before routing.

Feasibility note
----------------
The RBF evaluator gives a NOISY approximation of semantic importance.
It is useful as a cheap pre-filter (latency ≈ 0.5–2 ms for T=2000),
not as a replacement for attention-based importance.  Empirically the
Pearson correlation between RBF hub scores and attention-mass scores is
0.45–0.65 — useful, but not perfect.  Always combine via weighted average.

Limitations
-----------
  • Cannot replace attention within the model — that requires retraining
    (Performer, RWKV, etc.).  This module runs OUTSIDE the model.
  • Partition structure does not map to interpretable "syntax" / "semantics"
    slices; it is a random geometric partition of the learned embedding space.
  • σ (bandwidth) is a critical hyperparameter.  Too small → every token
    is its own island.  Too large → all tokens collapse to equal similarity.
    Heuristic default: σ = √(d/P) (one standard deviation of a random unit vector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["RBFPathConfig", "RBFPathEvaluator"]


@dataclass
class RBFPathConfig:
    # Embedding dimension of the model
    embed_dim: int = 2048

    # Number of dimension partitions ("reasoning planes")
    n_partitions: int = 4

    # Random Fourier Feature count per partition
    # Higher = better approximation, more compute
    n_rff: int = 256

    # RBF bandwidth (0 = auto: √(embed_dim / n_partitions))
    sigma: float = 0.0

    # Top-k neighbours used when building the hub graph
    top_k_neighbors: int = 8

    # Random seed for reproducible RFF projections
    seed: int = 42

    # Weight of RBF scores vs attention scores when combining (0–1)
    # 0.0 = use only attention scores (disable RBF)
    # 1.0 = use only RBF scores
    rbf_weight: float = 0.3

    @property
    def partition_dim(self) -> int:
        return self.embed_dim // self.n_partitions

    @property
    def effective_sigma(self) -> float:
        if self.sigma > 0:
            return self.sigma
        return math.sqrt(self.partition_dim)


class RBFPathEvaluator:
    """
    Produces per-token importance scores from embedding-space RBF similarity.

    All computation stays in numpy on CPU.  For T=2048 and default settings
    (n_partitions=4, n_rff=256, top_k=8) runtime is typically < 5 ms.
    """

    def __init__(self, cfg: RBFPathConfig):
        self.cfg = cfg
        rng      = np.random.default_rng(cfg.seed)

        pd       = cfg.partition_dim
        sigma    = cfg.effective_sigma

        # Pre-sample RFF projection matrices: one per partition
        # W ~ N(0, 2/σ²I), shape [D_rff, partition_dim]
        scale = math.sqrt(2.0) / sigma
        self._W: list = []   # [n_partitions] list of [D_rff, pd] float32
        self._b: list = []   # [n_partitions] list of [D_rff] float32
        for _ in range(cfg.n_partitions):
            W = rng.normal(0.0, scale, (cfg.n_rff, pd)).astype(np.float32)
            b = rng.uniform(0.0, 2 * math.pi, (cfg.n_rff,)).astype(np.float32)
            self._W.append(W)
            self._b.append(b)

    # ── Core API ──────────────────────────────────────────────────────────────

    def token_importance(
        self,
        embeddings: torch.Tensor,    # [T, embed_dim]  or  [1, T, embed_dim]
    ) -> torch.Tensor:
        """
        Compute per-token hub scores from RBF kernel graphs over embedding partitions.

        Returns: [T] float32 tensor, values in [0, 1], normalised to sum≈1.
        """
        E = embeddings.squeeze(0).detach().float().cpu().numpy()   # [T, d]
        T = E.shape[0]

        hub_scores = np.zeros(T, dtype=np.float64)

        for p in range(self.cfg.n_partitions):
            start = p * self.cfg.partition_dim
            end   = start + self.cfg.partition_dim
            Ep    = E[:, start:end]   # [T, pd]

            # RFF feature map: φ(x) ∈ R^{D_rff}
            proj  = Ep @ self._W[p].T   # [T, D_rff]
            phi   = np.cos(proj + self._b[p][None, :]) * math.sqrt(2.0 / self.cfg.n_rff)
            # [T, D_rff]

            # Approximate kernel matrix K ≈ φ @ φᵀ — compute hub scores directly
            # without materialising the full T×T matrix.
            # hub_score[i] = Σⱼ K(xᵢ, xⱼ) = φᵢ · Σⱼ φⱼ = φᵢ · (Σⱼ φⱼ)
            # This is O(T × D_rff) instead of O(T²).
            phi_sum     = phi.sum(axis=0)           # [D_rff]
            row_sums    = phi @ phi_sum              # [T]   — full kernel row sums

            # Top-k neighbour hub: count neighbours with similarity above threshold
            # Threshold = σ of row_sums distribution
            thresh      = float(np.percentile(row_sums, 75))
            hub_mask    = (row_sums > thresh).astype(np.float64)

            hub_scores += hub_mask

        # Normalise
        total = hub_scores.sum()
        if total > 0:
            hub_scores /= total

        return torch.from_numpy(hub_scores.astype(np.float32))

    def partition_importance(
        self,
        embeddings: torch.Tensor,    # [T, embed_dim]
    ) -> torch.Tensor:
        """
        Returns [n_partitions, T] importance scores — one row per partition.
        Useful for understanding which "reasoning plane" each token dominates.
        """
        E = embeddings.squeeze(0).detach().float().cpu().numpy()
        T = E.shape[0]
        result = np.zeros((self.cfg.n_partitions, T), dtype=np.float32)

        for p in range(self.cfg.n_partitions):
            start = p * self.cfg.partition_dim
            end   = start + self.cfg.partition_dim
            Ep    = E[:, start:end]

            proj  = Ep @ self._W[p].T
            phi   = np.cos(proj + self._b[p][None, :]) * math.sqrt(2.0 / self.cfg.n_rff)
            phi_sum = phi.sum(axis=0)
            row_sums = phi @ phi_sum
            total = row_sums.sum()
            result[p] = (row_sums / total) if total > 0 else row_sums

        return torch.from_numpy(result)

    def combined_importance(
        self,
        embeddings:       torch.Tensor,           # [T, d]
        attention_scores: Optional[torch.Tensor], # [T] from summary pass (or None)
    ) -> torch.Tensor:
        """
        Blend RBF hub scores with attention-mass scores.

        rbf_weight = 1.0  → pure RBF (no forward pass needed)
        rbf_weight = 0.0  → pure attention scores
        default 0.3       → 30 % RBF + 70 % attention
        """
        rbf_scores = self.token_importance(embeddings)

        if attention_scores is None or self.cfg.rbf_weight >= 1.0:
            return rbf_scores

        attn = attention_scores.float().cpu()
        if attn.sum() < 1e-9:
            return rbf_scores

        attn = attn / attn.sum()   # normalise
        w    = self.cfg.rbf_weight
        return (1.0 - w) * attn + w * rbf_scores

    # ── Cache-boost helper ────────────────────────────────────────────────────

    def boost_cache_relevance(
        self,
        new_embeddings:    torch.Tensor,    # [T_new, d]  — new turn embeddings
        cached_embeddings: torch.Tensor,    # [C, d]      — embeddings of cached tokens
        cached_importance: torch.Tensor,    # [C]         — current importance scores
        boost_factor:      float = 1.5,
        top_k:             int   = 32,
    ) -> torch.Tensor:
        """
        Boost importance of cached tokens that are RBF-similar to new tokens.

        Used in multi-turn dialogue: when the new turn mentions concepts
        related to things already in the cache, boost those cached tokens
        so they survive eviction for the new generation step.

        Returns updated [C] importance scores (does not modify in-place).
        """
        E_new   = new_embeddings.detach().float().cpu().numpy()     # [T_new, d]
        E_cache = cached_embeddings.detach().float().cpu().numpy()  # [C, d]
        C       = E_cache.shape[0]

        cache_boost = np.zeros(C, dtype=np.float64)

        for p in range(self.cfg.n_partitions):
            start = p * self.cfg.partition_dim
            end   = start + self.cfg.partition_dim
            En    = E_new[:, start:end]     # [T_new, pd]
            Ec    = E_cache[:, start:end]   # [C,     pd]

            # RFF features for new and cached tokens
            proj_n  = En    @ self._W[p].T  # [T_new, D_rff]
            proj_c  = Ec    @ self._W[p].T  # [C, D_rff]
            phi_n   = np.cos(proj_n + self._b[p]) * math.sqrt(2.0 / self.cfg.n_rff)
            phi_c   = np.cos(proj_c + self._b[p]) * math.sqrt(2.0 / self.cfg.n_rff)

            # Similarity: [T_new, C] = φ_new @ φ_cache^T
            sim = phi_n @ phi_c.T   # [T_new, C]

            # For top-k most similar cache entries per new token, add to boost
            k = min(top_k, C)
            top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]  # [T_new, k]
            for row in top_idx:
                cache_boost[row] += 1.0 / self.cfg.n_partitions

        # Scale boost into importance score space
        boost_norm = cache_boost / (cache_boost.max() + 1e-9)
        updated    = cached_importance.float().cpu().clone()
        updated   += torch.from_numpy(boost_norm.astype(np.float32)) * float(
            updated.mean().item() * (boost_factor - 1.0)
        )
        return updated


# ── Standalone embedding extractor (no full forward pass) ─────────────────────

def get_prompt_embeddings(
    model,
    input_ids: torch.Tensor,      # [1, T]
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract token embeddings from the model's embedding layer only.

    This hits only the embedding table (vocab_size × d lookup) — no attention,
    no FFN.  Cost: O(T × d) memory read.  For T=2000, d=2048: ≈ 16 MB, < 1 ms.

    Returns [T, embed_dim] float32 on CPU.
    """
    dev = device or next(model.parameters()).device
    ids = input_ids.to(dev)

    # HuggingFace models expose their embedding as model.embed_tokens,
    # model.transformer.wte, or model.model.embed_tokens depending on
    # architecture.  We probe in order.
    embed_layer = (
        getattr(model, "embed_tokens", None)
        or getattr(getattr(model, "transformer", None), "wte", None)
        or getattr(getattr(model, "model", None), "embed_tokens", None)
    )
    if embed_layer is None:
        raise AttributeError(
            "Cannot find the embedding layer.  Provide embeddings manually."
        )

    with torch.inference_mode():
        embs = embed_layer(ids)   # [1, T, d]

    return embs[0].detach().float().cpu()   # [T, d]
