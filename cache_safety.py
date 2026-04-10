"""
inference/cache_safety.py
--------------------------
Defensive guards for manual KV-cache manipulation.

When you step outside `model.generate()` and touch KV tensors directly,
a handful of silent failure modes appear.  This module centralises all
checks at the boundaries where they are needed:

  1. normalise_kv      — convert any HF cache object → legacy tuple-of-tuples
  2. safe_position_ids — clamp to model's max_position_embeddings
  3. validate_tokens   — filter out-of-vocab token IDs before injection
  4. ensure_dtype      — cast all KV tensors to a target dtype
  5. ensure_device     — move all KV tensors to a target device
  6. validate_mask     — verify causal mask shape matches KV fill
  7. check_model_hash  — warn when stored embeddings may be stale

Design
------
Each function is self-contained and raises ValueError or clips silently
depending on the severity.  Functions that can lose data always warn.

Usage
-----
Typical usage in two_pass.py right after model forward:

    out = model(ids, use_cache=True)
    pkv = normalise_kv(out.past_key_values)           # DynamicCache → tuple
    pkv = ensure_dtype(pkv, model.dtype)
    pkv = ensure_device(pkv, model.device)
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch

log = logging.getLogger(__name__)

__all__ = [
    "normalise_kv",
    "safe_position_ids",
    "validate_tokens",
    "ensure_dtype",
    "ensure_device",
    "ensure_kv_correct",
    "validate_mask",
    "check_model_hash",
    "kv_seq_len",
]

# Type alias for legacy KV format
LegacyKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


# ── 1. normalise_kv ──────────────────────────────────────────────────────────

# Try to import the base Cache class so we can do a reliable isinstance check.
# This covers DynamicCache, StaticCache, HybridCache, SlidingWindowCache, etc.
try:
    from transformers.cache_utils import Cache as _HFCacheBase
except ImportError:
    try:
        from transformers import Cache as _HFCacheBase
    except ImportError:
        _HFCacheBase = None


def normalise_kv(past_key_values) -> Optional[LegacyKV]:
    """
    Convert any HuggingFace KV-cache object to legacy tuple-of-(K,V)-tuples.

    Handles all known formats across transformers versions:
      - None                                → None
      - tuple / list of (K, V) pairs        → unchanged
      - transformers >= 4.36 Cache objects  → cache.layers[i].keys / .values
      - older DynamicCache (~4.26–4.35)     → key_cache / value_cache lists
      - very old transformers (< 4.26)      → to_legacy_cache()

    In transformers >= 4.36 the ``DynamicCache`` class was rewritten to use an
    internal ``self.layers`` list of ``CacheLayerMixin`` objects (``DynamicLayer``,
    ``DynamicSlidingWindowLayer``, etc.).  Each layer exposes ``.keys`` and
    ``.values`` tensors.  There is no longer a ``key_cache`` attribute or a working
    ``to_legacy_cache()`` method.

    Returns None for an empty (unfilled) cache.

    Raises
    ------
    TypeError  if the format is unrecognised and cannot be converted.
    """
    if past_key_values is None:
        return None

    # ── Path 0: already a legacy tuple / list of (K, V) pairs ────────────────
    if isinstance(past_key_values, (tuple, list)):
        if len(past_key_values) == 0:
            return None
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return tuple((item[0], item[1]) for item in past_key_values)
        raise TypeError(
            f"past_key_values is a {type(past_key_values).__name__} whose first "
            f"element has type {type(first).__name__} — expected a (K, V) pair."
        )

    # ── Path 1: transformers >= 4.36 — Cache.layers ───────────────────────────
    # DynamicCache, StaticCache, etc. store per-layer objects in self.layers.
    # Each layer (DynamicLayer / DynamicSlidingWindowLayer / StaticLayer) has
    # .keys and .values tensors set after the first forward pass.
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        try:
            result = tuple(
                (layer.keys, layer.values)
                for layer in layers
                if getattr(layer, "is_initialized", False)
                and getattr(layer, "keys", None) is not None
            )
            return result if result else None
        except Exception:
            pass  # unexpected structure; fall through

    # ── Path 2: older DynamicCache (~4.26-4.35) — key_cache / value_cache ────
    kc = getattr(past_key_values, "key_cache",   None)
    vc = getattr(past_key_values, "value_cache", None)
    if kc is not None and vc is not None:
        try:
            if len(kc) > 0:
                return tuple(zip(kc, vc))
            return None   # cache object exists but has no tokens yet
        except TypeError:
            pass

    # ── Path 3: to_legacy_cache() — very old transformers (< 4.26) ───────────
    to_legacy = getattr(past_key_values, "to_legacy_cache", None)
    if to_legacy is not None:
        try:
            result = to_legacy()
            if result is not None and len(result) > 0:
                return tuple(result)
        except Exception:
            pass

    # ── Path 4: __iter__ fallback ─────────────────────────────────────────────
    # DynamicCache.__iter__ (4.36+) yields (layer.keys, layer.values, optional)
    # — a 3-tuple.  We take just the first two elements per item.
    try:
        items = list(past_key_values)
        if items and isinstance(items[0], (tuple, list)) and len(items[0]) >= 2:
            return tuple((item[0], item[1]) for item in items)
    except Exception:
        pass

    raise TypeError(
        f"Cannot normalise past_key_values of type {type(past_key_values).__name__}. "
        "Expected a DynamicCache (transformers>=4.36), or legacy tuple/list of (K, V) pairs."
    )


# ── 2. kv_seq_len ─────────────────────────────────────────────────────────────

def kv_seq_len(past_key_values) -> int:
    """
    Return the sequence length encoded in a (possibly unnormalised) KV cache.
    Works with DynamicCache, StaticCache, legacy tuples, or None.
    """
    if past_key_values is None:
        return 0

    # Official DynamicCache method
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return past_key_values.get_seq_length()
        except Exception:
            pass

    # Attribute access
    if hasattr(past_key_values, "key_cache"):
        kc = past_key_values.key_cache
        if kc and hasattr(kc[0], "shape"):
            return kc[0].shape[2]
        return 0

    # Legacy tuple
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and hasattr(first[0], "shape"):
            return first[0].shape[2]

    return 0


# ── 3. safe_position_ids ─────────────────────────────────────────────────────

def safe_position_ids(
    positions:       torch.Tensor,       # [N] integer positions
    model_max_length: int,
    warn:            bool = True,
) -> torch.Tensor:
    """
    Clamp all positions to [0, model_max_length - 1].

    Prevents IndexError in RoPE when session-restored positions are beyond
    the model's max_position_embeddings.
    """
    clipped = positions.clamp(0, model_max_length - 1)
    if warn and not torch.equal(clipped, positions):
        n_clipped = (positions != clipped).sum().item()
        warnings.warn(
            f"safe_position_ids: clipped {n_clipped} position IDs that exceeded "
            f"model_max_length={model_max_length}. "
            "Session continuity may be slightly degraded.",
            stacklevel=2,
        )
    return clipped


# ── 4. validate_tokens ────────────────────────────────────────────────────────

def validate_tokens(
    token_ids:     List[int],
    vocab_size:    int,
    unk_token_id:  int = 0,
) -> List[int]:
    """
    Replace out-of-vocabulary token IDs with `unk_token_id`.

    Protects against loading stale SQLite sessions that contain token IDs
    from a different tokenizer version or a larger vocabulary model.
    """
    safe = []
    n_replaced = 0
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            safe.append(tid)
        else:
            safe.append(unk_token_id)
            n_replaced += 1
    if n_replaced:
        warnings.warn(
            f"validate_tokens: replaced {n_replaced}/{len(token_ids)} out-of-vocab token IDs "
            f"(vocab_size={vocab_size}). Session may have been stored by a different model.",
            stacklevel=2,
        )
    return safe


# ── 5. ensure_dtype ───────────────────────────────────────────────────────────

def ensure_dtype(
    past_key_values: LegacyKV,
    dtype:           torch.dtype,
) -> LegacyKV:
    """
    Cast all K/V tensors in a legacy tuple to `dtype`.

    Prevents "expected dtype Half but found Float" errors when mixing
    float16 caches with float32 model output or vice versa.
    """
    if past_key_values is None:
        return None
    return tuple(
        (k.to(dtype=dtype), v.to(dtype=dtype))
        for k, v in past_key_values
    )


# ── 6. ensure_device ──────────────────────────────────────────────────────────

def ensure_device(
    past_key_values: LegacyKV,
    device:          Union[str, torch.device],
) -> LegacyKV:
    """
    Move all K/V tensors in a legacy tuple to `device`.

    Prevents "Expected all tensors to be on the same device" errors when
    tensors come from SQLite retrieval (CPU) and need to go into a GPU cache.
    """
    if past_key_values is None:
        return None
    return tuple(
        (k.to(device=device), v.to(device=device))
        for k, v in past_key_values
    )


# ── 7. ensure_kv_correct (combined) ──────────────────────────────────────────

def ensure_kv_correct(
    past_key_values,
    device: Union[str, torch.device],
    dtype:  torch.dtype,
) -> Optional[LegacyKV]:
    """
    One-call wrapper: normalise format, cast dtype, move to device.

    Typical use right after model forward:

        pkv = ensure_kv_correct(out.past_key_values, model.device, model.dtype)
    """
    pkv = normalise_kv(past_key_values)
    if pkv is None:
        return None
    pkv = ensure_dtype(pkv, dtype)
    pkv = ensure_device(pkv, device)
    return pkv


# ── 8. validate_mask ─────────────────────────────────────────────────────────

def validate_mask(
    attention_mask:  torch.Tensor,     # [1, 1, Q, K] or [1, K]
    cache_seq_len:   int,
    query_len:       int = 1,
) -> torch.Tensor:
    """
    Trim or pad an attention mask so its key dimension matches `cache_seq_len`.

    Prevents shape mismatch errors when the cache is pruned (tokens removed)
    but the mask still reflects the original sequence length.

    Returns a mask of shape [..., query_len, cache_seq_len].
    """
    if attention_mask is None:
        return attention_mask

    if attention_mask.dim() == 2:
        # [1, seq_len] — simple 1D mask
        current_k = attention_mask.shape[-1]
        if current_k == cache_seq_len:
            return attention_mask
        if current_k > cache_seq_len:
            return attention_mask[:, -cache_seq_len:]
        # Pad left with zeros (masked out)
        pad = torch.zeros(
            attention_mask.shape[0], cache_seq_len - current_k,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        return torch.cat([pad, attention_mask], dim=-1)

    if attention_mask.dim() == 4:
        # [1, 1, Q, K] — 4D causal mask (typically float, -inf for masked)
        current_q = attention_mask.shape[-2]
        current_k = attention_mask.shape[-1]

        # Fix K dimension
        if current_k > cache_seq_len:
            attention_mask = attention_mask[..., :, -cache_seq_len:]
        elif current_k < cache_seq_len:
            pad = torch.full(
                (*attention_mask.shape[:-1], cache_seq_len - current_k),
                float("-inf"),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)

        return attention_mask

    return attention_mask   # Unknown layout — pass through


# ── 9. check_model_hash ───────────────────────────────────────────────────────

def check_model_hash(
    model,
    stored_hash: Optional[str],
    warn_on_mismatch: bool = True,
) -> str:
    """
    Compute a lightweight fingerprint of the model (first + last weight tensor
    shapes + dtype).  Compare against `stored_hash` to detect if the current
    model differs from the one that generated stored embeddings.

    Returns the current hash string (store it alongside embeddings).
    """
    try:
        params = list(model.named_parameters())
        if not params:
            return "unknown"
        # Use first and last parameter shapes + dtype as a cheap fingerprint
        first_name, first_p = params[0]
        last_name,  last_p  = params[-1]
        fingerprint = (
            f"{first_name}|{tuple(first_p.shape)}|{first_p.dtype}|"
            f"{last_name}|{tuple(last_p.shape)}|{last_p.dtype}"
        )
        current_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:16]
        if stored_hash is not None and stored_hash != current_hash and warn_on_mismatch:
            warnings.warn(
                f"check_model_hash: current model hash ({current_hash!r}) differs from "
                f"stored hash ({stored_hash!r}). Session embeddings may be stale — "
                "FAISS retrieval quality may be degraded.",
                stacklevel=2,
            )
        return current_hash
    except Exception:
        return "unknown"
