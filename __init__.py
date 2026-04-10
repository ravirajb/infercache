"""
inference/
----------
Hierarchical KV-cache inference primitives for long-context LLM generation.

Public API
----------
  HierarchicalInferenceEngine   — top-level engine (recommended entry point)
  EngineConfig                  — all knobs in one dataclass
  build_qwen_config             — auto-detect config from a Qwen model name

  Lower-level primitives (for custom pipelines):
  HierarchicalKVCache           — 3-level fixed-budget KV cache
  CacheConfig                   — cache dimensions / budget
  TwoPassEngine                 — two-pass inference logic
  MultiDomainCache              — per-domain cache banks
  DomainRouter                  — lightweight prompt classifier
  RBFPathEvaluator              — embedding-space importance scoring
  AttentionKLLSketch            — streaming quantile sketch for eviction

Quick start
-----------
  from inference import HierarchicalInferenceEngine, build_qwen_config

  cfg = build_qwen_config("Qwen/Qwen2.5-3B-Instruct")
  eng = HierarchicalInferenceEngine.from_pretrained(
      "Qwen/Qwen2.5-3B-Instruct", cfg
  )
  print(eng.generate("Explain RBF kernels in two paragraphs."))
"""

from .kll_sketch         import AttentionKLLSketch
from .cache_safety       import (
    normalise_kv, ensure_kv_correct, ensure_dtype, ensure_device,
    safe_position_ids, validate_tokens, validate_mask, check_model_hash,
    kv_seq_len,
)
from .hierarchical_cache import CacheConfig, CacheStats, HierarchicalKVCache
from .two_pass           import TwoPassConfig, TwoPassEngine, SummaryResult
from .domain_cache       import Domain, DomainRouter, MultiDomainCache
from .rbf_path           import RBFPathConfig, RBFPathEvaluator, get_prompt_embeddings
from .engine             import EngineConfig, HierarchicalInferenceEngine, build_qwen_config
from .graph_cache        import ClusterConfig, GraphKVCache, MultiClusterCache
from .triton_ops         import (
    gather_cache_kv, scatter_importance, topk_min_slot,
    compact_sdp_attention, GPUImportanceTracker,
)
from .faiss_index        import FaissAugmentedCache, FaissIVFIndex, faiss_available
from .benchmark          import BenchmarkSuite, BenchmarkResult
from .code_cache         import (
    code_engine_config, CodeTokenScorer, IdentifierPinner,
)
from .colab_t4           import build_engine, check_gpu, triton_warmup, interactive_stream
from .multiturn_bench    import MultiTurnBench, MultiTurnResult, TurnResult, print_report
from .session_store      import SessionStore, SessionRecord, TurnRecord, TokenRecord
from .paged_cache        import (
    PAGE_SIZE, PagePool, KVAllocator, MultiSessionCache,
    PagedSessionView, PagedCacheConfig,
)
from .memory_engine      import PersistentMemoryEngine, SessionConfig, GenerationResult

__all__ = [
    # Top-level entry point
    "HierarchicalInferenceEngine",
    "EngineConfig",
    "build_qwen_config",
    # Cache safety
    "normalise_kv",
    "ensure_kv_correct",
    "ensure_dtype",
    "ensure_device",
    "safe_position_ids",
    "validate_tokens",
    "validate_mask",
    "check_model_hash",
    "kv_seq_len",
    # Cache
    "HierarchicalKVCache",
    "CacheConfig",
    "CacheStats",
    # Two-pass
    "TwoPassEngine",
    "TwoPassConfig",
    "SummaryResult",
    # Domain routing
    "MultiDomainCache",
    "DomainRouter",
    "Domain",
    # RBF
    "RBFPathEvaluator",
    "RBFPathConfig",
    "get_prompt_embeddings",
    # Sketch
    "AttentionKLLSketch",
    # Graph cache
    "ClusterConfig",
    "GraphKVCache",
    "MultiClusterCache",
    # Triton ops
    "gather_cache_kv",
    "scatter_importance",
    "topk_min_slot",
    "compact_sdp_attention",
    "GPUImportanceTracker",
    # FAISS
    "FaissAugmentedCache",
    "FaissIVFIndex",
    "faiss_available",
    # Benchmark
    "BenchmarkSuite",
    "BenchmarkResult",
    # Code-specific optimizations
    "code_engine_config",
    "CodeTokenScorer",
    "IdentifierPinner",
    # Colab / T4 runner
    "build_engine",
    "check_gpu",
    "triton_warmup",
    "interactive_stream",
    # Multi-turn benchmark
    "MultiTurnBench",
    "MultiTurnResult",
    "TurnResult",
    "print_report",
    # SQLite3 session store
    "SessionStore",
    "SessionRecord",
    "TurnRecord",
    "TokenRecord",
    # Paged KV cache
    "PAGE_SIZE",
    "PagePool",
    "KVAllocator",
    "MultiSessionCache",
    "PagedSessionView",
    "PagedCacheConfig",
    # Persistent memory engine
    "PersistentMemoryEngine",
    "SessionConfig",
    "GenerationResult",
]
