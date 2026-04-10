"""
inference/session_store.py
---------------------------
SQLite3-backed session memory store.

What is persisted
-----------------
For every turn in every session we store, per domain cache slot:

  token_id   (int)          — the original vocabulary token ID
  position   (int)          — its absolute sequence position
  importance (float32)      — accumulated importance score at end of turn
  embedding  (BLOB)         — last-layer hidden state, float16, shape [embed_dim]

What is NOT persisted
---------------------
Raw KV tensors (too large and device-specific).  On session resume we
re-run a single forward pass on the retrieved token IDs at their original
positions to produce fresh, device-compatible KV state.  For Qwen 3B, a
128-token forward pass takes ~5 ms on T4 — well within the 100 ms budget.

Schema
------
  sessions (session_id PK, user_id, domain, created_at, updated_at)
  session_turns (turn_id AI, session_id FK, turn_idx, prompt, response,
                 n_prompt_tokens, n_response_tokens, elapsed_ms, created_at)
  token_memory (id AI, session_id FK, turn_idx, token_id, position,
                importance, embedding BLOB)

FAISS in-memory index
---------------------
SessionStore maintains one faiss.IndexFlatIP per session_id in RAM.
The index is rebuilt on first query_similar() call and kept warm across
subsequent calls.  Index size: n_tokens × embed_dim × float32.
For 512 tokens, 2048-dim float32: 4 MB per session — negligible.
Index is evicted when the session has not been accessed for `index_ttl_s`.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

_FAISS = False
try:
    import faiss
    _FAISS = True
except ImportError:
    pass

__all__ = [
    "SessionStore",
    "SessionRecord",
    "TurnRecord",
    "TokenRecord",
]

# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    session_id: str
    user_id:    str
    domain:     str        = "LANGUAGE"
    created_at: float      = 0.0
    updated_at: float      = 0.0


@dataclass
class TurnRecord:
    session_id:        str
    turn_idx:          int
    prompt:            str
    response:          str
    n_prompt_tokens:   int   = 0
    n_response_tokens: int   = 0
    elapsed_ms:        float = 0.0


class TokenRecord(NamedTuple):
    token_id:   int
    position:   int
    importance: float
    embedding:  np.ndarray   # float16 [embed_dim]


# ── SQLite schema ─────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-32000;

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT    PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    domain      TEXT    DEFAULT 'LANGUAGE',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turns (
    turn_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL REFERENCES sessions(session_id),
    turn_idx          INTEGER NOT NULL,
    prompt            TEXT    NOT NULL,
    response          TEXT    NOT NULL,
    n_prompt_tokens   INTEGER DEFAULT 0,
    n_response_tokens INTEGER DEFAULT 0,
    elapsed_ms        REAL    DEFAULT 0,
    created_at        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS token_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
    turn_idx    INTEGER NOT NULL,
    token_id    INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    importance  REAL    NOT NULL,
    embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_token_memory_session
    ON token_memory(session_id, importance DESC);
CREATE INDEX IF NOT EXISTS idx_session_turns_session
    ON session_turns(session_id, turn_idx);
"""


# ── Embedding serialisation ───────────────────────────────────────────────────

def _emb_to_blob(emb: np.ndarray) -> bytes:
    """float16 ndarray → BLOB (shape prefix + data)."""
    arr = emb.astype(np.float16)
    # header: uint32 ndim, then uint32 per dim
    header = struct.pack(f">I{'I'*arr.ndim}", arr.ndim, *arr.shape)
    return header + arr.tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    """BLOB → float16 ndarray."""
    ndim   = struct.unpack(">I", blob[:4])[0]
    shape  = struct.unpack(f">{'I'*ndim}", blob[4:4 + 4*ndim])
    data   = blob[4 + 4*ndim:]
    return np.frombuffer(data, dtype=np.float16).reshape(shape).copy()


# ── Session store ─────────────────────────────────────────────────────────────

class SessionStore:
    """
    SQLite3-backed persistent session store with optional FAISS retrieval.

    Thread-safe via per-connection threading (check_same_thread=False +
    WAL journal mode for concurrent readers).

    Parameters
    ----------
    db_path         : file path for the SQLite database
                      (use ":memory:" for testing)
    max_tokens_per_session : max token_memory rows kept per session.
                      Oldest / lowest-importance rows are pruned on save.
    index_ttl_s     : seconds before an in-RAM FAISS index is evicted
    """

    def __init__(
        self,
        db_path:                str   = "session_memory.db",
        max_tokens_per_session: int   = 512,
        index_ttl_s:            float = 600.0,
    ) -> None:
        self.db_path     = db_path
        self.max_tokens  = max_tokens_per_session
        self.index_ttl_s = index_ttl_s

        self._local      = threading.local()
        self._lock       = threading.Lock()
        # {session_id: (faiss_index, np_matrix, last_access_time)}
        self._faiss_cache: Dict[str, tuple] = {}

        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """One connection per thread (SQLite requirement)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(_DDL)

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def upsert_session(
        self,
        session_id: str,
        user_id:    str,
        domain:     str = "LANGUAGE",
    ) -> SessionRecord:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, domain, created_at, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at""",
                (session_id, user_id, domain, now, now),
            )
        return SessionRecord(session_id, user_id, domain, now, now)

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        return SessionRecord(**dict(row)) if row else None

    def list_sessions(self, user_id: str) -> List[SessionRecord]:
        rows = self._conn().execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [SessionRecord(**dict(r)) for r in rows]

    # ── Turn CRUD ─────────────────────────────────────────────────────────────

    def save_turn(self, rec: TurnRecord) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO session_turns
                   (session_id, turn_idx, prompt, response,
                    n_prompt_tokens, n_response_tokens, elapsed_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (rec.session_id, rec.turn_idx, rec.prompt, rec.response,
                 rec.n_prompt_tokens, rec.n_response_tokens,
                 rec.elapsed_ms, time.time()),
            )

    def get_turns(
        self, session_id: str, last_n: int = 10
    ) -> List[TurnRecord]:
        rows = self._conn().execute(
            """SELECT * FROM session_turns
               WHERE session_id=?
               ORDER BY turn_idx DESC LIMIT ?""",
            (session_id, last_n),
        ).fetchall()
        return [
            TurnRecord(
                session_id        = r["session_id"],
                turn_idx          = r["turn_idx"],
                prompt            = r["prompt"],
                response          = r["response"],
                n_prompt_tokens   = r["n_prompt_tokens"],
                n_response_tokens = r["n_response_tokens"],
                elapsed_ms        = r["elapsed_ms"],
            )
            for r in reversed(rows)
        ]

    # ── Token memory write ────────────────────────────────────────────────────

    def save_token_memory(
        self,
        session_id:  str,
        turn_idx:    int,
        token_ids:   List[int],
        positions:   List[int],
        importances: List[float],
        embeddings:  Optional[np.ndarray] = None,   # [N, embed_dim] float16
    ) -> None:
        """
        Write token memory for one turn.

        If `embeddings` is None the BLOB column is left as NULL —
        query_similar() will still work but return random results.

        After writing, prune the session to max_tokens_per_session rows
        keeping the highest-importance tokens.
        """
        now = time.time()
        rows = []
        for i, (tid, pos, imp) in enumerate(zip(token_ids, positions, importances)):
            blob = _emb_to_blob(embeddings[i]) if embeddings is not None else None
            rows.append((session_id, turn_idx, int(tid), int(pos), float(imp), blob))

        with self._tx() as conn:
            conn.executemany(
                """INSERT INTO token_memory
                   (session_id, turn_idx, token_id, position, importance, embedding)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            # Prune: keep top-N by importance
            conn.execute(
                """DELETE FROM token_memory
                   WHERE session_id=? AND id NOT IN (
                       SELECT id FROM token_memory
                       WHERE session_id=?
                       ORDER BY importance DESC
                       LIMIT ?
                   )""",
                (session_id, session_id, self.max_tokens),
            )

        # Invalidate cached FAISS index for this session
        with self._lock:
            self._faiss_cache.pop(session_id, None)

    # ── Token memory read ─────────────────────────────────────────────────────

    def load_token_memory(
        self,
        session_id: str,
        top_k:      int = 256,
    ) -> List[TokenRecord]:
        """
        Return the top-k most important token records for a session.
        Returned in ascending position order (chronological).
        """
        rows = self._conn().execute(
            """SELECT token_id, position, importance, embedding
               FROM token_memory
               WHERE session_id=?
               ORDER BY importance DESC LIMIT ?""",
            (session_id, top_k),
        ).fetchall()
        records = []
        for r in rows:
            emb = _blob_to_emb(r["embedding"]) if r["embedding"] else np.zeros(1, dtype=np.float16)
            records.append(TokenRecord(r["token_id"], r["position"], r["importance"], emb))
        # Sort by position
        records.sort(key=lambda x: x.position)
        return records

    # ── FAISS retrieval ───────────────────────────────────────────────────────

    def _get_faiss_index(self, session_id: str):
        """Build (or return cached) a FAISS IndexFlatIP for the session."""
        with self._lock:
            entry = self._faiss_cache.get(session_id)
            if entry and (time.time() - entry[2]) < self.index_ttl_s:
                return entry[0], entry[1]  # (index, emb_matrix)

        records = self.load_token_memory(session_id, top_k=self.max_tokens)
        if not records or not _FAISS:
            return None, None

        emb_dim = records[0].embedding.shape[0]
        matrix  = np.stack([r.embedding for r in records]).astype(np.float32)  # [N, D]

        # Unit-normalise for cosine similarity via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
        matrix /= norms

        index = faiss.IndexFlatIP(emb_dim)
        index.add(matrix)

        with self._lock:
            self._faiss_cache[session_id] = (index, matrix, time.time())

        return index, matrix

    def query_similar(
        self,
        session_id:    str,
        query_emb:     np.ndarray,    # [embed_dim] float32
        top_k:         int = 64,
    ) -> List[TokenRecord]:
        """
        Return the top_k session tokens most similar to `query_emb`.
        Falls back to load_token_memory(top_k) if FAISS is unavailable.
        """
        records = self.load_token_memory(session_id, top_k=self.max_tokens)
        if not records:
            return []

        if not _FAISS:
            # Fallback: return top_k by importance
            return records[:top_k]

        index, matrix = self._get_faiss_index(session_id)
        if index is None:
            return records[:top_k]

        q = query_emb.astype(np.float32).reshape(1, -1)
        q /= max(np.linalg.norm(q), 1e-8)

        top_k_actual = min(top_k, len(records))
        _, idx       = index.search(q, top_k_actual)   # idx: [1, K]
        return [records[i] for i in idx[0] if 0 <= i < len(records)]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def session_turn_count(self, session_id: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) AS n FROM session_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return row["n"] if row else 0

    def delete_session(self, session_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM token_memory  WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM session_turns WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions      WHERE session_id=?", (session_id,))
        with self._lock:
            self._faiss_cache.pop(session_id, None)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
