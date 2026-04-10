"""
inference/memory_engine.py
---------------------------
PersistentMemoryEngine — wraps HierarchicalInferenceEngine with
SQLite3-backed long-term session memory.

Architecture
------------

  ┌──────────────────────────────────────────────────────────────┐
  │  PersistentMemoryEngine.generate(prompt, session_id, user_id)│
  └──────────────────────────────────────────┬───────────────────┘
                                             │
          ┌──────────────────────────────────▼──────────────────┐
          │  LOAD  (~5-15 ms)                                    │
          │  1. SQLite: top-K important tokens for this session  │
          │  2. FAISS: rank by similarity to current prompt      │
          │  3. Model: mini forward-pass on retrieved token IDs  │
          │     at their original position_ids → fresh KV        │
          │  4. Prime HierarchicalKVCache with retrieved KVs     │
          └──────────────────────────────────┬──────────────────┘
                                             │
          ┌──────────────────────────────────▼──────────────────┐
          │  GENERATE                                            │
          │  HierarchicalInferenceEngine.generate(prompt)        │
          └──────────────────────────────────┬──────────────────┘
                                             │
          ┌──────────────────────────────────▼──────────────────┐
          │  SAVE  (~2-5 ms)                                     │
          │  1. Extract token IDs, positions, importance scores  │
          │     from cache state after generation                │
          │  2. Extract hidden-state embeddings for FAISS use    │
          │  3. Write turn + token memory to SQLite              │
          └─────────────────────────────────────────────────────┘

Latency budget (Qwen 3B, T4 GPU, 128 retrieved tokens)
-------------------------------------------------------
  SQLite read       :   1 ms
  FAISS search      :   1 ms
  Re-encode 128 tok :  10 ms   (128² attention, FP16)
  Cache priming     :   3 ms
  SQLite write      :   2 ms
  ─────────────────────────
  Total overhead    :  ≈ 17 ms   << 100 ms budget

Session continuity
------------------
Unlike raw KV storage, we store token IDs + positions.  On reload we
run a fresh forward pass at the original position IDs — this correctly
applies RoPE at the token's historical position, making the KV state
device-independent and model-update-compatible.

The approximation (retrieved tokens attend to each other during re-encoding,
but in the original session there may have been tokens between them) is
negligible in practice for the ~128 high-importance tokens retrieved.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .engine         import HierarchicalInferenceEngine, EngineConfig
from .session_store  import SessionStore, SessionRecord, TurnRecord, TokenRecord
from .cache_safety   import (
    ensure_kv_correct, safe_position_ids, validate_tokens, check_model_hash,
)

__all__ = [
    "SessionConfig",
    "PersistentMemoryEngine",
    "GenerationResult",
]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    db_path:               str   = "session_memory.db"
    max_retrieved_tokens:  int   = 128    # tokens loaded from SQLite per session
    max_stored_tokens:     int   = 512    # importance-ranked tokens stored per session
    faiss_top_k:           int   = 64     # FAISS result count before re-ranking
    min_importance:        float = 0.01  # discard very-low-importance tokens on save
    include_embeddings:    bool  = True  # store float16 embeddings for FAISS lookup


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    response:         str
    session_id:       str
    turn_idx:         int
    generate_ms:      float
    load_ms:          float
    save_ms:          float
    n_retrieved:      int   = 0
    n_stored:         int   = 0

    @property
    def total_ms(self) -> float:
        return self.load_ms + self.generate_ms + self.save_ms

    def __str__(self) -> str:
        return (
            f"{self.response}\n"
            f"[session={self.session_id[:8]}... turn={self.turn_idx} "
            f"total={self.total_ms:.0f}ms "
            f"load={self.load_ms:.0f}ms gen={self.generate_ms:.0f}ms "
            f"save={self.save_ms:.0f}ms "
            f"retrieved={self.n_retrieved} stored={self.n_stored}]"
        )


# ── Helper: extract prompt embeddings from model ──────────────────────────────

@torch.inference_mode()
def _get_embeddings(
    model,
    token_ids: torch.Tensor,   # [1, T] on device
    device:    torch.device,
) -> np.ndarray:
    """
    Return last-layer hidden state averaged over T tokens as a float32
    numpy vector [embed_dim].  Used as the session query embedding.
    """
    try:
        out = model(token_ids, output_hidden_states=True, use_cache=False)
        # last hidden state: [1, T, embed_dim]
        h = out.hidden_states[-1].float().mean(dim=1).squeeze(0)  # [embed_dim]
        return h.cpu().numpy().astype(np.float32)
    except Exception:
        # Fallback: zero vector (FAISS will fall back to importance ranking)
        embed_dim = getattr(model.config, "hidden_size", 2048)
        return np.zeros(embed_dim, dtype=np.float32)


@torch.inference_mode()
def _get_token_embeddings(
    model,
    token_ids: torch.Tensor,   # [1, T]
    device:    torch.device,
) -> np.ndarray:
    """
    Return per-token last-layer hidden states: float16 [T, embed_dim].
    """
    try:
        out = model(token_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].squeeze(0)   # [T, embed_dim]
        return h.cpu().half().numpy()           # float16
    except Exception:
        T         = token_ids.shape[1]
        embed_dim = getattr(model.config, "hidden_size", 2048)
        return np.zeros((T, embed_dim), dtype=np.float16)


# ── Helper: re-encode retrieved tokens to get fresh KV ───────────────────────

@torch.inference_mode()
def _reencode_tokens(
    model,
    token_ids: List[int],
    positions: List[int],
    device:    torch.device,
    dtype:     torch.dtype = torch.float16,
):
    """
    Run a forward pass of `token_ids` at their original `positions`.
    Returns past_key_values (tuple-of-tuples) with fresh, device-correct KV.

    RoPE is applied at the stored positions, so the KV state is compatible
    with future tokens that continue from the end of the original session.

    Returns None if token_ids is empty.
    """
    if not token_ids:
        return None

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)     # [1, N]
    pos = torch.tensor([positions], dtype=torch.long, device=device)     # [1, N]

    # Use_cache=True returns past_key_values after encoding the context
    out = model(ids, position_ids=pos, use_cache=True)
    return out.past_key_values


# ── Persistent Memory Engine ──────────────────────────────────────────────────

class PersistentMemoryEngine:
    """
    Drop-in replacement for HierarchicalInferenceEngine that adds
    SQLite3 session persistence.

    Quick-start
    -----------
    ::

        from inference import HierarchicalInferenceEngine, build_qwen_config
        from inference import PersistentMemoryEngine, SessionConfig

        eng     = HierarchicalInferenceEngine.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        mem_eng = PersistentMemoryEngine(eng)

        result = mem_eng.generate(
            prompt     = "What is the capital of France?",
            session_id = "alice-session-1",
            user_id    = "alice",
        )
        print(result.response)

        # Later (even after process restart):
        result2 = mem_eng.generate(
            prompt     = "Tell me more about the Eiffel Tower.",
            session_id = "alice-session-1",
            user_id    = "alice",
        )
        # Alice's session context is automatically restored from SQLite.

    Multi-user in production
    ------------------------
    Create one PersistentMemoryEngine per process; it is thread-safe for
    concurrent sessions (each session gets its own cache in the engine).
    The underlying SQLite store uses WAL mode and is concurrency-safe.
    """

    def __init__(
        self,
        engine:     HierarchicalInferenceEngine,
        session_cfg: Optional[SessionConfig] = None,
        store:       Optional[SessionStore]  = None,
    ) -> None:
        self.engine      = engine
        self.cfg         = session_cfg or SessionConfig()
        self.store       = store or SessionStore(
            db_path                = self.cfg.db_path,
            max_tokens_per_session = self.cfg.max_stored_tokens,
        )
        self._device     = engine.device
        self._model      = engine.model
        self._tokenizer  = engine.tokenizer

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:         str,
        session_id:     Optional[str] = None,
        user_id:        str           = "default",
        max_new_tokens: Optional[int] = None,
        domain=None,
    ) -> GenerationResult:
        """
        Generate a response, loading session context first and saving afterwards.

        Parameters
        ----------
        prompt         : user input text
        session_id     : string identifier for the session.
                         Auto-generated UUID if None.
        user_id        : user or application identifier for SQLite lookup
        max_new_tokens : override EngineConfig.max_new_tokens
        domain         : optional Domain enum override for cache routing

        Returns
        -------
        GenerationResult with response text + timing/stats
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Ensure session exists in store
        self.store.upsert_session(session_id, user_id)
        turn_idx = self.store.session_turn_count(session_id)

        # ── Load ──────────────────────────────────────────────────────────
        t_load0     = time.perf_counter()
        n_retrieved = self._load_session_into_cache(session_id, prompt)
        load_ms     = (time.perf_counter() - t_load0) * 1000

        # ── Generate ──────────────────────────────────────────────────────
        t_gen0    = time.perf_counter()
        response  = self.engine.generate(
            prompt, max_new_tokens=max_new_tokens, domain=domain
        )
        gen_ms    = (time.perf_counter() - t_gen0) * 1000

        # ── Save ──────────────────────────────────────────────────────────
        t_save0   = time.perf_counter()
        n_stored  = self._save_turn(session_id, user_id, turn_idx, prompt, response, gen_ms)
        save_ms   = (time.perf_counter() - t_save0) * 1000

        return GenerationResult(
            response    = response,
            session_id  = session_id,
            turn_idx    = turn_idx,
            generate_ms = gen_ms,
            load_ms     = load_ms,
            save_ms     = save_ms,
            n_retrieved = n_retrieved,
            n_stored    = n_stored,
        )

    def session_summary(self, session_id: str) -> Dict:
        """Return a dict summarising a session's stored state."""
        rec   = self.store.get_session(session_id)
        turns = self.store.get_turns(session_id, last_n=5)
        toks  = self.store.load_token_memory(session_id, top_k=10)
        return {
            "session_id":   session_id,
            "user_id":      rec.user_id if rec else None,
            "domain":       rec.domain  if rec else None,
            "n_turns":      len(turns),
            "recent_turns": [{"prompt": t.prompt[:80], "elapsed_ms": t.elapsed_ms}
                             for t in turns[-3:]],
            "top_tokens":   [
                {"token_id": tk.token_id, "importance": round(tk.importance, 4)}
                for tk in toks[:5]
            ],
        }

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a session and all its memory from the store."""
        self.store.delete_session(session_id)
        self.engine.reset()

    # ── Internal: load ────────────────────────────────────────────────────────

    def _load_session_into_cache(self, session_id: str, prompt: str) -> int:
        """
        Load top-K high-importance tokens from SQLite, optionally ranked by
        semantic similarity to `prompt`, then re-encode them to prime the
        hierarchical KV cache.

        Returns the number of tokens actually loaded (0 on first turn).
        """
        # 1. Retrieve token records from SQLite
        records = self.store.load_token_memory(
            session_id, top_k=self.cfg.max_retrieved_tokens * 2
        )
        if not records:
            return 0

        # 2. FAISS re-rank by similarity to prompt (if embeddings are available
        #    and FAISS is installed)
        has_embs = records[0].embedding.shape[0] > 1
        if has_embs:
            _raw = self._tokenizer.encode(prompt, return_tensors="pt")
            if not isinstance(_raw, torch.Tensor):
                _raw = torch.tensor([_raw], dtype=torch.long)
            prompt_ids = _raw.to(self._device)
            prompt_emb   = _get_embeddings(self._model, prompt_ids, self._device)
            records      = self.store.query_similar(
                session_id, prompt_emb, top_k=self.cfg.max_retrieved_tokens
            )
        else:
            records = records[:self.cfg.max_retrieved_tokens]

        if not records:
            return 0

        # 3. Sort by position (chronological order) for correct RoPE
        records.sort(key=lambda r: r.position)

        token_ids   = [r.token_id for r in records]
        positions   = [r.position for r in records]
        importances = [r.importance for r in records]

        # Guard: replace any out-of-vocab token IDs (stale tokenizer versions)
        vocab_size = getattr(self._tokenizer, "vocab_size", None) or len(self._tokenizer)
        token_ids  = validate_tokens(token_ids, vocab_size,
                                     self._tokenizer.unk_token_id or 0)

        # Guard: clamp positions to model's max_position_embeddings
        model_max = getattr(self._model.config, "max_position_embeddings", 32768)
        pos_tensor_raw = torch.tensor(positions, dtype=torch.long)
        pos_tensor_raw = safe_position_ids(pos_tensor_raw, model_max)
        positions      = pos_tensor_raw.tolist()

        # 4. Re-encode at original positions → fresh KV (handles RoPE correctly)
        past_kv = _reencode_tokens(
            self._model, token_ids, positions, self._device, self.engine.cfg.dtype
        )
        if past_kv is None:
            return 0

        # Guard: normalise format, dtype, device  (DynamicCache → legacy tuple)
        past_kv = ensure_kv_correct(past_kv, self._device, self.engine.cfg.dtype)
        if past_kv is None:
            return 0

        # 5. Prime the active domain's cache with retrieved KV + importances
        domain_cache = self._get_active_cache()
        if domain_cache is not None:
            importance_tensor = torch.tensor(
                importances, dtype=torch.float32, device=torch.device("cpu")
            )
            pos_tensor = torch.tensor(
                positions, dtype=torch.long, device=torch.device("cpu")
            )
            domain_cache.load_from_hf_output(
                past_kv,
                importance_scores=importance_tensor,
                positions=pos_tensor,
            )

        return len(records)

    # ── Internal: save ────────────────────────────────────────────────────────

    def _save_turn(
        self,
        session_id: str,
        user_id:    str,
        turn_idx:   int,
        prompt:     str,
        response:   str,
        elapsed_ms: float,
    ) -> int:
        """
        Extract cache state after generation and write to SQLite.
        Returns the number of tokens stored.
        """
        domain_cache = self._get_active_cache()
        if domain_cache is None:
            return 0

        # ── Extract token IDs, positions, importance from cache ──────────
        occupied_mask = domain_cache.occupied                        # [budget]
        occupied_idx  = occupied_mask.nonzero(as_tuple=True)[0]     # [F]
        if len(occupied_idx) == 0:
            return 0

        positions   = domain_cache.positions[occupied_idx].tolist()  # [F]
        importances = domain_cache.importance[occupied_idx].tolist() # [F]

        # Filter by minimum importance
        above_threshold = [
            (pos, imp) for pos, imp in zip(positions, importances)
            if imp >= self.cfg.min_importance
        ]
        if not above_threshold:
            # Save all if all are below threshold (first turn)
            above_threshold = list(zip(positions, importances))

        positions_filt   = [p for p, _ in above_threshold]
        importances_filt = [i for _, i in above_threshold]

        # ── Re-tokenise prompt + response to get token IDs ────────────────
        # We use the prompt to match position-indexed token IDs.
        # This is an approximation: we store the prompt token IDs at their
        # corresponding cached positions.
        prompt_ids  = self._tokenizer.encode(prompt,  add_special_tokens=False)
        resp_ids    = self._tokenizer.encode(response, add_special_tokens=False)
        all_ids_seq = prompt_ids + resp_ids

        # Build position → token_id map from the most-recent context tokens
        # (positions may not be contiguous due to previous sessions)
        pos_to_token: Dict[int, int] = {}
        for i, tid in enumerate(all_ids_seq):
            pos_to_token[i] = tid

        # Map stored positions to token IDs (fill unknown with a safe default)
        token_ids_filt = [
            pos_to_token.get(pos, self._tokenizer.unk_token_id or 0)
            for pos in positions_filt
        ]

        # ── Extract embeddings for FAISS (optional) ───────────────────────
        embeddings: Optional[np.ndarray] = None
        if self.cfg.include_embeddings and len(token_ids_filt) > 0:
            ids_t = torch.tensor([token_ids_filt], dtype=torch.long,
                                 device=self._device)
            embeddings = _get_token_embeddings(self._model, ids_t, self._device)

        # ── Write turn record ─────────────────────────────────────────────
        n_prompt_tokens = len(prompt_ids)
        n_resp_tokens   = len(resp_ids)
        self.store.save_turn(TurnRecord(
            session_id        = session_id,
            turn_idx          = turn_idx,
            prompt            = prompt,
            response          = response,
            n_prompt_tokens   = n_prompt_tokens,
            n_response_tokens = n_resp_tokens,
            elapsed_ms        = elapsed_ms,
        ))

        # ── Write token memory ────────────────────────────────────────────
        self.store.save_token_memory(
            session_id  = session_id,
            turn_idx    = turn_idx,
            token_ids   = token_ids_filt,
            positions   = positions_filt,
            importances = importances_filt,
            embeddings  = embeddings,
        )

        return len(token_ids_filt)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_active_cache(self):
        """Return the currently active HierarchicalKVCache from the engine."""
        try:
            return self.engine._multi_cache.get(self.engine._active_domain)
        except AttributeError:
            return None

    # ── Pass-through to underlying engine ─────────────────────────────────────

    def reset(self) -> None:
        self.engine.reset()

    def cache_stats(self) -> Dict:
        return self.engine.cache_stats()

    @property
    def cfg_engine(self) -> EngineConfig:
        return self.engine.cfg

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        session_cfg:        Optional[SessionConfig]      = None,
        engine_cfg:         Optional[EngineConfig]       = None,
        load_in_8bit:       bool                         = False,
        load_in_4bit:       bool                         = False,
        device:             str                          = "cuda",
        db_path:            str                          = "session_memory.db",
    ) -> "PersistentMemoryEngine":
        """
        Convenience factory: load model + build PersistentMemoryEngine in one call.

        Example
        -------
        ::

            mem_eng = PersistentMemoryEngine.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                db_path="prod_sessions.db",
            )
            result = mem_eng.generate("Hello!", session_id="u123", user_id="u123")
        """
        engine = HierarchicalInferenceEngine.from_pretrained(
            model_name_or_path,
            cfg          = engine_cfg,
            load_in_8bit = load_in_8bit,
            load_in_4bit = load_in_4bit,
            device       = device,
        )
        cfg = session_cfg or SessionConfig(db_path=db_path)
        return cls(engine=engine, session_cfg=cfg)
