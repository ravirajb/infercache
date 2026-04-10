"""
inference/two_pass.py
---------------------
Two-pass inference protocol for long prompts.

Why two passes?
  Standard auto-regressive LLM generation on a long prompt is O(T²) in
  the prompt length T — the KV cache grows with every generated token and
  the attention cost per step grows proportionally.

  Two-pass reasoning:
    PASS 1 — Summary (O(T²) of standard attention, unavoidable for first pass)
      Run the full prompt through the model with output_attentions=True.
      Collect the per-token attention mass (how much each token was attended to).
      This tells us WHICH tokens are semantically critical for generation.

    PASS 2 — Generation (O(K) per step, constant in T)
      Populate the hierarchical cache with the top-K "heavy hitter" tokens
      (attention sinks + recent window + important pool).
      Generate auto-regressively attending only to these K cached tokens.
      Memory cost is O(K × num_layers × …) regardless of T.

Routing:
  T ≤ summary_threshold → skip Pass 1, use direct generation (standard KV).
  T >  summary_threshold → two-pass.

Attention-weight interpretation (H2O heavy-hitter principle)
-------------------------------------------------------------
  Column-sum of the attention matrix gives the "attention mass received" by
  each token.  A token at column j receives:
    received[j] = Σ_h Σ_i A[h, i, j]
  High received score = other tokens strongly depend on this one = heavy hitter.
  Empirically, <20 % of tokens accumulate >80 % of attention mass (power-law
  distribution), so keeping 20-30 % of tokens preserves most quality while
  cutting context length to a fixed budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .hierarchical_cache import CacheConfig, HierarchicalKVCache, _normalise_past_kv
from .cache_safety        import ensure_kv_correct

__all__ = ["TwoPassConfig", "SummaryResult", "TwoPassEngine"]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TwoPassConfig:
    # Threshold: use two-pass only when prompt is longer than this.
    # 128 is the right default for multi-turn: after 2-3 turns the accumulated
    # history crosses this, engaging the hierarchical cache.  512 was too high
    # — short prompts always routed to _generate_direct, bypassing the cache.
    summary_threshold:  int   = 128

    # Generation hyper-parameters
    max_new_tokens:     int   = 256
    temperature:        float = 1.0
    top_p:              float = 0.95
    top_k:              int   = 0    # 0 = disabled
    repetition_penalty: float = 1.0

    # ── Importance source for the summary pass ────────────────────────────
    # use_hidden_importance = True  (DEFAULT, RECOMMENDED for T4 / int8 / int4)
    #   Use the L2 norm of each token's last-layer hidden state as importance.
    #   • Works with ANY attention backend (sdpa, flash_attn_2, eager).
    #   • ~3-5× faster than attention-based importance (no O(T²) attn matrix).
    #   • Quality is comparable; activation magnitude is a well-established
    #     proxy for token importance in pruning literature.
    #
    # use_hidden_importance = False  (original, research-accurate)
    #   Collect output_attentions=True from the model (requires eager backend)
    #   and compute column-sum attention mass per token (H2O principle).
    #   Use only when the model is loaded with attn_implementation='eager'.
    use_hidden_importance: bool = True

    # Only used when use_hidden_importance=False:
    # True  = sum attention over ALL layers (best quality, high memory)
    # False = last layer only (faster)
    all_layers_importance: bool = False
    importance_layer: int = -1


# ── Summary pass result ───────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    """Output from the summary pass."""
    importance_scores: torch.Tensor   # [seq_len]  float32, CPU
    past_key_values:   object         # raw HF past_key_values
    last_logits:       torch.Tensor   # [vocab_size]  logits of last token
    seq_len:           int
    hidden_states:     Optional[torch.Tensor] = None  # [1, T, D] CPU — optional


# ── Token importance computation ──────────────────────────────────────────────

def _attention_mass(
    attentions: Tuple,             # tuple of [1, H, T, T] per layer
    all_layers: bool,
    layer_idx:  int,
) -> torch.Tensor:
    """
    Compute per-token attention-received scores from HF attention tensors.

    Each layer attention tensor A has shape [batch=1, heads, seq, seq].
    A[0, h, i, j] = weight placed on token j by query position i, head h.

    column_sum[j] = Σ_h Σ_i A[0, h, i, j]
      → how much total attention was received by token j across all heads/queries.

    We accumulate across layers (if all_layers=True) and normalise by
    total mass to get a probability-like importance score in [0, 1].
    """
    layers = attentions if all_layers else (attentions[layer_idx % len(attentions)],)

    importance: Optional[torch.Tensor] = None
    for attn in layers:
        # attn: [1, H, T, T]
        # Sum over batch (1), heads (H), query (T) to get received mass per key
        received = attn[0].sum(dim=(0, 1))   # [T]
        importance = received if importance is None else importance + received

    if importance is None:
        # Fallback — should never happen
        T = attentions[0].shape[-1]
        return torch.ones(T, dtype=torch.float32)

    importance = importance.float().cpu()
    total = importance.sum()
    return importance / total if total > 1e-9 else importance


# ── Main engine ───────────────────────────────────────────────────────────────

class TwoPassEngine:
    """
    Wraps any HuggingFace CausalLM with two-pass inference.

    Designed to be model-agnostic; tested with Qwen2/Qwen2.5 3B.

    Typical usage
    -------------
    cfg_cache = CacheConfig(num_layers=28, num_kv_heads=8, head_dim=128)
    engine = TwoPassEngine(model, tokenizer, cfg_cache)
    ids = engine.generate_ids("Explain Fourier transforms.", max_new_tokens=200)
    print(tokenizer.decode(ids))
    """

    def __init__(
        self,
        model,
        tokenizer,
        cache_cfg:  CacheConfig,
        cfg:        Optional[TwoPassConfig] = None,
        device:     str = "cuda",
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cache_cfg = cache_cfg
        self.cfg       = cfg or TwoPassConfig()
        self.device    = torch.device(device)

        self._cache    = HierarchicalKVCache(cache_cfg, device=self.device)

    # ── Pass 1: summary ───────────────────────────────────────────────────────

    @torch.inference_mode()
    def summary_pass(self, input_ids: torch.Tensor) -> SummaryResult:
        """
        Full forward pass over the prompt. Computes per-token importance scores.

        Two modes controlled by cfg.use_hidden_importance:

        Hidden-state mode (default, fast, SDPA-compatible)
        ---------------------------------------------------
        Forward with output_hidden_states=True only.
        Importance[i] = ||h_last[i]||_2 / Σ_j ||h_last[j]||_2
        Cost: O(T × D) — no attention matrix materialised.
        Works with sdpa, flash_attn_2, eager.

        Attention-mass mode (slower, needs eager backend)
        --------------------------------------------------
        Forward with output_attentions=True.
        Importance[i] = column-sum of attention matrix (H2O principle).
        Cost: O(T²) — full attention matrix per layer.
        Requires attn_implementation='eager' at model load time.
        """
        ids = input_ids.to(self.device)

        if self.cfg.use_hidden_importance:
            # ── Fast path: hidden-state L2 importance ─────────────────────
            out = self.model(
                ids,
                output_attentions  = False,
                output_hidden_states = True,
                use_cache          = True,
            )
            # Last encoder layer hidden states: [1, T, D]
            h = out.hidden_states[-1][0].float()   # [T, D]
            importance = h.norm(dim=-1)             # [T]
            total = importance.sum()
            importance = (importance / total if total > 1e-9 else importance).cpu()
            hidden = out.hidden_states[-1].detach().cpu()
        else:
            # ── Original path: attention-column-sum importance ─────────────
            out = self.model(
                ids,
                output_attentions     = True,
                output_hidden_states  = self.cfg.all_layers_importance,
                use_cache             = True,
            )
            importance = _attention_mass(
                out.attentions,
                self.cfg.all_layers_importance,
                self.cfg.importance_layer,
            )
            hidden = (out.hidden_states[-1].detach().cpu()
                      if out.hidden_states is not None else None)

        # Store past_key_values as-is (DynamicCache / legacy tuple).
        # _prime_cache → load_from_hf_output → _normalise_past_kv converts it
        # to legacy format right before use.  _write_kv handles dtype/device.
        return SummaryResult(
            importance_scores = importance,
            past_key_values   = out.past_key_values,
            last_logits       = out.logits[0, -1].detach().cpu(),
            seq_len           = ids.shape[1],
            hidden_states     = hidden,
        )

    # ── Cache priming from summary result ─────────────────────────────────────

    def _prime_cache(self, summary: SummaryResult) -> None:
        """Load the hierarchical cache from the summary-pass KV snapshot."""
        self._cache.reset()
        self._cache.load_from_hf_output(
            past_key_values   = summary.past_key_values,
            importance_scores = summary.importance_scores,
            positions         = torch.arange(summary.seq_len, dtype=torch.long),
        )

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor, generated: List[int]) -> int:
        """Single-step sampling with temperature, top-p, top-k, and rep-penalty."""
        cfg = self.cfg
        logits = logits.float().clone()

        # Repetition penalty
        if cfg.repetition_penalty != 1.0 and generated:
            for tok in set(generated):
                if logits[tok] > 0:
                    logits[tok] /= cfg.repetition_penalty
                else:
                    logits[tok] *= cfg.repetition_penalty

        if cfg.temperature == 0.0 or cfg.temperature < 1e-4:
            return int(logits.argmax(-1).item())

        logits /= cfg.temperature

        if cfg.top_k > 0:
            vals = logits.topk(cfg.top_k).values
            logits[logits < vals[-1]] = float("-inf")

        if cfg.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cum = sorted_logits.softmax(-1).cumsum(-1)
            remove = (cum - sorted_logits.softmax(-1)) > cfg.top_p
            logits[sorted_idx[remove]] = float("-inf")

        return int(logits.softmax(-1).multinomial(1).item())

    # ── Generation — two-pass path ────────────────────────────────────────────

    @torch.inference_mode()
    def _generate_two_pass(
        self,
        input_ids:    torch.Tensor,   # [1, T]
        eos_token_id: Optional[int],
    ) -> List[int]:
        """
        Generate tokens using the two-pass protocol:
          1. Summary pass → prime hierarchical cache.
          2. Auto-regressive decode using compact cache (~O(K) per step).
        """
        summary = self.summary_pass(input_ids)
        self._prime_cache(summary)

        generated = []
        # Prime first logits from the last token of Pass-1
        logits = summary.last_logits.to(self.device)
        T      = summary.seq_len

        for step in range(self.cfg.max_new_tokens):
            next_id = self._sample(logits, generated)
            generated.append(next_id)

            if eos_token_id is not None and next_id == eos_token_id:
                break
            if len(generated) >= self.cfg.max_new_tokens:
                break

            # Build compact past_kv from current cache state
            past_kv = self._cache.to_hf_past_key_values(device=self.device)
            cur_pos  = T + step
            pos_ids  = torch.tensor([[cur_pos]], device=self.device, dtype=torch.long)

            out = self.model(
                torch.tensor([[next_id]], device=self.device),
                past_key_values=past_kv,
                position_ids=pos_ids,
                use_cache=True,
            )
            logits = out.logits[0, -1]

            # Extract this step's K/V and push into hierarchical cache.
            # _normalise_past_kv delegates to the robust cache_safety.normalise_kv.
            new_pkv = _normalise_past_kv(out.past_key_values)
            n_layers = len(new_pkv)
            # new_pkv[li][0] shape: [1, H, prev_fill+1, D] — we want the last slot
            new_k = torch.stack(
                [new_pkv[li][0][0, :, -1, :] for li in range(n_layers)], dim=0
            ).unsqueeze(2)     # [L, H, 1, D]
            new_v = torch.stack(
                [new_pkv[li][1][0, :, -1, :] for li in range(n_layers)], dim=0
            ).unsqueeze(2)

            self._cache.push(new_k, new_v, position=cur_pos + 1)

        return generated

    # ── Generation — direct path (short prompts) ──────────────────────────────

    @torch.inference_mode()
    def _generate_direct(
        self,
        input_ids:    torch.Tensor,
        eos_token_id: Optional[int],
    ) -> List[int]:
        """
        Standard KV-cache generation for prompts at or below the threshold.
        No summary pass.  Past KV grows unboundedly but is short by assumption.
        """
        generated = []
        past_kv   = None
        cur_ids   = input_ids.to(self.device)

        for step in range(self.cfg.max_new_tokens):
            out = self.model(cur_ids, past_key_values=past_kv, use_cache=True)
            logits  = out.logits[0, -1]
            next_id = self._sample(logits, generated)
            generated.append(next_id)

            if eos_token_id is not None and next_id == eos_token_id:
                break

            # The model created this cache and expects it back as-is.
            # Converting to legacy tuple is unnecessary here (and fails on
            # newer transformers where DynamicCache is not subscriptable).
            past_kv = out.past_key_values
            cur_ids = torch.tensor([[next_id]], device=self.device)

        return generated

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_ids(
        self,
        prompt:       str,
        max_new_tokens: Optional[int] = None,
        eos_token_id:   Optional[int] = None,
    ) -> List[int]:
        """
        Tokenise `prompt` and generate up to max_new_tokens token IDs.

        Routing:
          len(tokens) > summary_threshold → two-pass (O(K) per decode step)
          len(tokens) ≤ summary_threshold → direct (standard KV grows with steps)
        """
        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        ids = self.tokenizer.encode(prompt, return_tensors="pt")   # [1, T]
        T   = ids.shape[1]

        if T > self.cfg.summary_threshold:
            return self._generate_two_pass(ids, eos_token_id)
        else:
            return self._generate_direct(ids, eos_token_id)

    def generate(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper — returns decoded text."""
        ids = self.generate_ids(prompt, **kwargs)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def stream_generate(
        self,
        prompt:       str,
        eos_token_id: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Stream-decode: yield decoded token strings one at a time.

        Note: For two-pass path the summary pass runs fully before any token
        is yielded (first-token latency = summary pass time ≈ one forward pass).
        """
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        T   = ids.shape[1]

        if T > self.cfg.summary_threshold:
            gen_ids = self._generate_two_pass(ids, eos_token_id)
        else:
            gen_ids = self._generate_direct(ids, eos_token_id)

        for tok in gen_ids:
            yield self.tokenizer.decode([tok], skip_special_tokens=False)

    def reset_cache(self) -> None:
        """Clear the hierarchical cache.  Call between independent conversations."""
        self._cache.reset()
