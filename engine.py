"""
inference/engine.py
--------------------
HierarchicalInferenceEngine — top-level assembly.

Combines all inference primitives into a single, usable API:

  Primitives used
  ---------------
  TwoPassEngine      fresh-prompt two-pass inference (O(K) per decode step)
  MultiDomainCache   domain-specific KV cache banks (code / Q&A / language)
  RBFPathEvaluator   embedding-space pre-filter for cache boost & context compression
  HierarchicalKVCache fixed-budget 3-level KV cache (L0 sink / L1 recent / L2 important)

  Model adapter
  -------------
  Currently tested with: Qwen2 / Qwen2.5 3B (and any Qwen2-architecture model).
  Design is model-agnostic: any HuggingFace AutoModelForCausalLM is accepted,
  provided it:
    • supports output_attentions=True (some flash-attention variants do not)
    • accepts past_key_values as tuple-of-tuples or DynamicCache
    • exposes embed_tokens or model.embed_tokens for RBF pre-filtering

Complexity summary
------------------
  First prompt, T ≤ threshold:  O(T²) attention (standard), O(T) per decode step
  First prompt, T >  threshold:  O(T²) summary pass (once), O(K) per decode step
  Subsequent prompts (same session, cache warm):
    With RBF boost:              O(T × d) embedding lookup + O(T × D_rff), then O(K) decode
    Without RBF:                 O(K) per decode step immediately
  
  K = total_budget (sink + recent + important), a constant.
  "O(1) per decode step in context length" is achieved once the cache is primed.

Quick-start example
-------------------
  from inference import HierarchicalInferenceEngine, EngineConfig, build_qwen_config

  cfg = build_qwen_config("Qwen/Qwen2.5-3B-Instruct")
  eng = HierarchicalInferenceEngine.from_pretrained(
      "Qwen/Qwen2.5-3B-Instruct", cfg
  )
  print(eng.generate("Explain transformer attention in simple terms."))
  print(eng.generate("Now give me Python code that computes softmax."))  # domain switch → CODE cache
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .hierarchical_cache import CacheConfig, HierarchicalKVCache
from .domain_cache        import Domain, DomainRouter, MultiDomainCache
from .two_pass            import TwoPassConfig, TwoPassEngine, SummaryResult
from .rbf_path            import RBFPathConfig, RBFPathEvaluator, get_prompt_embeddings

__all__ = [
    "EngineConfig",
    "build_qwen_config",
    "HierarchicalInferenceEngine",
]


# ── Engine configuration ──────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Top-level configuration that drives all sub-components."""

    # ── Cache budget (per domain, so memory × n_domains) ──────────────────
    sink_size:      int = 4      # L0 — attention sink tokens
    recent_size:    int = 256    # L1 — recent sliding window
    important_size: int = 252    # L2 — KLL-selected important tokens
    evict_quantile: float = 0.15
    cache_decay:    float = 0.99

    # ── Two-pass control ──────────────────────────────────────────────────
    summary_threshold:     int   = 512   # tokens — above this → two-pass
    max_new_tokens:        int   = 256
    temperature:           float = 1.0
    top_p:                 float = 0.95
    top_k:                 int   = 0
    repetition_penalty:    float = 1.0
    all_layers_importance: bool  = False  # False = last layer only (faster; True needs eager)

    # ── Summary-pass importance mode ─────────────────────────────────────
    # True  = hidden-state L2 norm (fast, sdpa/flash compatible, default)
    # False = attention-column-sum (H2O, requires eager backend, slower)
    use_hidden_importance: bool = True

    # ── RBF pre-filter ────────────────────────────────────────────────────
    use_rbf:        bool  = True   # disable to skip RBF entirely
    rbf_partitions: int   = 4
    rbf_n_rff:      int   = 256
    rbf_sigma:      float = 0.0    # 0 = auto
    rbf_weight:     float = 0.3    # blend weight (0 = pure attn, 1 = pure RBF)

    # ── Model architecture — required for cache allocation ─────────────────
    num_layers:     int = 28     # e.g., Qwen2.5-3B has 36; Qwen2-1.5B has 28
    num_kv_heads:   int = 8      # GQA KV heads
    head_dim:       int = 128
    embed_dim:      int = 2048

    # ── Runtime ───────────────────────────────────────────────────────────
    device: str = "cuda"
    dtype:  torch.dtype = torch.float16

    def cache_config(self) -> CacheConfig:
        return CacheConfig(
            num_layers     = self.num_layers,
            num_kv_heads   = self.num_kv_heads,
            head_dim       = self.head_dim,
            sink_size      = self.sink_size,
            recent_size    = self.recent_size,
            important_size = self.important_size,
            evict_quantile = self.evict_quantile,
            decay          = self.cache_decay,
            dtype          = self.dtype,
        )

    def two_pass_config(self) -> TwoPassConfig:
        return TwoPassConfig(
            summary_threshold      = self.summary_threshold,
            max_new_tokens         = self.max_new_tokens,
            temperature            = self.temperature,
            top_p                  = self.top_p,
            top_k                  = self.top_k,
            repetition_penalty     = self.repetition_penalty,
            use_hidden_importance  = self.use_hidden_importance,
            all_layers_importance  = self.all_layers_importance,
        )

    def rbf_config(self) -> RBFPathConfig:
        return RBFPathConfig(
            embed_dim    = self.embed_dim,
            n_partitions = self.rbf_partitions,
            n_rff        = self.rbf_n_rff,
            sigma        = self.rbf_sigma,
            rbf_weight   = self.rbf_weight,
        )


def build_qwen_config(model_name_or_path: str, device: str = "cuda") -> EngineConfig:
    """
    Auto-detect architecture dimensions from a Qwen model config and return
    a matching EngineConfig.  Falls back to Qwen2.5-3B defaults on error.

    Supports: Qwen2-1.5B, Qwen2-7B, Qwen2.5-3B, Qwen2.5-7B, and later models.
    """
    defaults = {
        # model_size_hint → (num_layers, num_kv_heads, head_dim, embed_dim)
        "1.5b":  (28,  8,  64, 1536),
        "3b":    (36,  8, 128, 2048),
        "7b":    (28,  8, 128, 3584),
        "14b":   (48,  8, 128, 5120),
        "32b":   (64,  8, 128, 5120),
    }

    try:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        num_layers   = getattr(hf_cfg, "num_hidden_layers", 36)
        num_kv_heads = getattr(hf_cfg, "num_key_value_heads", 8)
        head_dim     = getattr(hf_cfg, "head_dim",
                               getattr(hf_cfg, "hidden_size", 2048) //
                               getattr(hf_cfg, "num_attention_heads", 16))
        embed_dim    = getattr(hf_cfg, "hidden_size", 2048)
    except Exception:
        # Fallback to 3B defaults
        num_layers, num_kv_heads, head_dim, embed_dim = defaults["3b"]

    return EngineConfig(
        num_layers   = num_layers,
        num_kv_heads = num_kv_heads,
        head_dim     = head_dim,
        embed_dim    = embed_dim,
        device       = device,
    )


# ── Main engine ───────────────────────────────────────────────────────────────

class HierarchicalInferenceEngine:
    """
    Production inference engine assembling all inference/ primitives.

    Lifecycle
    ---------
    1. Construct: HierarchicalInferenceEngine.from_pretrained(model_name, cfg)
    2. First turn: engine.generate(prompt)
       → Routes to domain cache → Two-pass if T > threshold
    3. Subsequent turns: engine.generate(next_prompt)
       → Domain cache already warm → RBF boost → O(K) decode
    4. Domain switch: engine.route_domain(new_prompt) — explicit override
    5. Session end:   engine.reset()

    Memory layout (approximate, Qwen2.5-3B, budget=512, 3 domains, fp16):
      3 × (28 × 8 × 512 × 128 × 2 bytes) ≈ 88 MB
      Compare: standard KV at 8K context:
      28 × 8 × 8192 × 128 × 2 bytes ≈ 4.7 GB
      Savings: ~53 × at 8K context.  Grows with context length.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cfg: EngineConfig,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        # ── Sub-components ────────────────────────────────────────────────
        cache_cfg   = cfg.cache_config()
        two_pass_cfg = cfg.two_pass_config()

        self._multi_cache = MultiDomainCache(cache_cfg, device=self.device)
        self._router      = DomainRouter()

        # Two-pass engine shares whatever cache we point it to
        # We'll rebind the internal cache per domain before generating
        self._two_pass = TwoPassEngine(
            model     = model,
            tokenizer = tokenizer,
            cache_cfg = cache_cfg,
            cfg       = two_pass_cfg,
            device    = cfg.device,
        )

        # RBF evaluator (optional)
        self._rbf: Optional[RBFPathEvaluator] = None
        if cfg.use_rbf:
            self._rbf = RBFPathEvaluator(cfg.rbf_config())

        # Tracking
        self._turn_count:     int = 0
        self._active_domain:  Domain = Domain.LANGUAGE
        self._timing:         Dict[str, float] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cfg: Optional[EngineConfig] = None,
        load_in_8bit:   bool = False,
        load_in_4bit:   bool = False,
        device:         str  = "cuda",
    ) -> "HierarchicalInferenceEngine":
        """
        Load a HuggingFace CausalLM model and tokenizer, then construct
        a HierarchicalInferenceEngine around them.

        Tested with: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2-1.5B-Instruct
        Should work with: any Qwen2*, LLaMA-3*, Mistral, Phi-3 model.

        Note: output_attentions=True is incompatible with BetterTransformer
        and xformers attention backends.  If you use those, set
        all_layers_importance=False (uses only the last layer's weights).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg is None:
            cfg = build_qwen_config(model_name_or_path, device=device)

        print(f"Loading tokenizer: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        print(f"Loading model: {model_name_or_path}")
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Choose the fastest available attention backend.
        # 'eager' is only needed when use_hidden_importance=False (attention-mass
        # summary pass), which requires output_attentions=True.  For the default
        # hidden-state importance mode we can use the much faster 'sdpa'.
        _use_hidden = getattr(cfg, "use_hidden_importance", True) if cfg else True
        if not _use_hidden:
            _attn_impl = "eager"   # output_attentions=True needs eager
        else:
            # Flash-attention 2 is fastest; fall back to sdpa (fused on T4 via cuBLAS)
            try:
                import flash_attn  # noqa: F401
                _attn_impl = "flash_attention_2"
            except ImportError:
                _attn_impl = "sdpa"

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype            = torch.float16 if device == "cuda" else torch.float32,
            device_map             = device,
            trust_remote_code      = True,
            quantization_config    = quantization_config,
            attn_implementation    = _attn_impl,
        )
        print(f"  Attention backend: {_attn_impl}")
        model.eval()
        print(f"  Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")

        return cls(model, tokenizer, cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:             str,
        max_new_tokens:     Optional[int]   = None,
        temperature:        Optional[float] = None,
        top_p:              Optional[float] = None,
        domain:             Optional[Domain] = None,   # override auto-routing
    ) -> str:
        """
        Generate a response to `prompt`, using the full hierarchical pipeline.

        First turn:        two-pass if T > threshold
        Subsequent turns:  warm cache lookup (O(K) per decode step)
        """
        ids = self.generate_ids(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, domain=domain,
        )
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate_ids(
        self,
        prompt:         str,
        max_new_tokens: Optional[int]   = None,
        temperature:    Optional[float] = None,
        top_p:          Optional[float] = None,
        domain:         Optional[Domain] = None,
    ) -> List[int]:
        """Generate token IDs for `prompt`."""
        t0 = time.perf_counter()

        # ── Step 1: Domain routing ─────────────────────────────────────────
        active_domain = domain if domain is not None else self._router.route(prompt)
        self._active_domain = active_domain
        domain_cache = self._multi_cache.get(active_domain)

        # Point TwoPassEngine at the domain cache
        self._two_pass._cache = domain_cache

        # Override generation params if provided
        tp_cfg = self._two_pass.cfg
        if max_new_tokens is not None:
            tp_cfg.max_new_tokens = max_new_tokens
        if temperature is not None:
            tp_cfg.temperature = temperature
        if top_p is not None:
            tp_cfg.top_p = top_p

        # ── Step 2: Tokenise ──────────────────────────────────────────────
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        T = input_ids.shape[1]

        # ── Step 3: RBF pre-filter for warm cache (subsequent turns) ──────
        if self._rbf is not None and self._turn_count > 0 and domain_cache.current_fill > 0:
            self._rbf_boost(input_ids, domain_cache)

        # ── Step 4: Generate ──────────────────────────────────────────────
        if T > self.cfg.summary_threshold:
            generated = self._two_pass._generate_two_pass(
                input_ids, self.tokenizer.eos_token_id
            )
        else:
            generated = self._two_pass._generate_direct(
                input_ids, self.tokenizer.eos_token_id
            )

        self._turn_count += 1
        self._timing["last_generate_ms"] = (time.perf_counter() - t0) * 1000
        return generated

    def stream_generate(
        self,
        prompt:         str,
        max_new_tokens: Optional[int]   = None,
        domain:         Optional[Domain] = None,
    ) -> Iterator[str]:
        """Streaming version — yields decoded text fragments."""
        gen_ids = self.generate_ids(
            prompt, max_new_tokens=max_new_tokens, domain=domain
        )
        for tok in gen_ids:
            yield self.tokenizer.decode([tok], skip_special_tokens=False)

    # ── Cache management ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all domain caches and reset turn counter."""
        self._multi_cache.reset_all()
        self._turn_count = 0
        self._active_domain = Domain.LANGUAGE

    def reset_domain(self, domain: Optional[Domain] = None) -> None:
        """Clear only the specified domain's cache (default = active domain)."""
        self._multi_cache.reset_domain(domain or self._active_domain)

    def cache_stats(self) -> Dict[str, object]:
        """Return fill-level stats for all domain caches."""
        return self._multi_cache.stats()

    # ── RBF boost helper ──────────────────────────────────────────────────────

    def _rbf_boost(
        self,
        input_ids:    torch.Tensor,
        domain_cache: HierarchicalKVCache,
    ) -> None:
        """
        Boost the importance of cached tokens that are semantically close
        to the new prompt's embedding.  Called on warm-cache turns.

        Cost: O(T_new × D_rff × n_partitions + C × D_rff × n_partitions)
        where C = cache fill (constant ≤ budget).
        """
        try:
            new_embs = get_prompt_embeddings(self.model, input_ids, self.device)
        except AttributeError:
            return   # Model doesn't expose embed layer — skip silently

        # We need embeddings for cached tokens too.
        # Use their stored positions as a proxy to extract embeddings.
        cached_pos = domain_cache.get_position_ids_for_cache()
        if cached_pos.shape[0] == 0:
            return

        # Clamp to valid range: positions may exceed original input
        max_pos = input_ids.shape[1] - 1
        valid   = cached_pos[cached_pos <= max_pos]
        if valid.shape[0] == 0:
            return

        try:
            valid_ids = input_ids[0, valid].unsqueeze(0)   # [1, C_valid]
            cached_embs = get_prompt_embeddings(self.model, valid_ids, self.device)
        except (AttributeError, IndexError):
            return

        # Get current importance for valid slots
        occ          = domain_cache.occupied.nonzero(as_tuple=True)[0]
        sorted_occ   = occ[domain_cache.positions[occ].argsort()]
        valid_count  = min(valid.shape[0], cached_embs.shape[0])
        valid_slots  = sorted_occ[:valid_count]

        updated_imp = self._rbf.boost_cache_relevance(
            new_embeddings    = new_embs,
            cached_embeddings = cached_embs[:valid_count],
            cached_importance = domain_cache.importance[valid_slots],
            boost_factor      = 1.5,
            top_k             = 32,
        )
        domain_cache.importance[valid_slots] = updated_imp.to(domain_cache.importance.device)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def timing(self) -> Dict[str, float]:
        """Return last-call timing breakdown in milliseconds."""
        return dict(self._timing)

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"HierarchicalInferenceEngine("
            f"budget={cfg.sink_size+cfg.recent_size+cfg.important_size}, "
            f"threshold={cfg.summary_threshold}, "
            f"domains=[CODE,QA,LANGUAGE], "
            f"rbf={'on' if self._rbf else 'off'}, "
            f"device={cfg.device})"
        )
