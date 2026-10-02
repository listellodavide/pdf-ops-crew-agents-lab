"""In-process stand-in for the Azure AI Search and Foundry memory-store calls the labs use.

Used by the unit tests and as the contingency when a team cannot reach Azure on the day
(PDFMEM_BACKEND=simulator). It is NOT the architecture being taught: the resident memory lives
in the cloud. Behaviour follows the documented service semantics closely enough for the
exercises (OData filters, BM25, kNN with scalar/binary quantization and rescoring, hybrid RRF,
scoring profiles), but numbers such as index sizes and latency are estimates.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pdfmem.text import bm25, minmax

_INDEXES: dict[str, dict] = {}          # process-wide, like a service
_STORES: dict[str, dict] = {}
_loaded = False


def _state_path() -> Path:
    return Path(os.environ.get("PDFMEM_MEMORY", "memory")) / ".simulator.pkl"


def _load() -> None:
    """The simulator persists between CLI runs (like the real service) in memory/.simulator.pkl."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    path = _state_path()
    if path.exists():
        indexes, stores = pickle.loads(path.read_bytes())
        _INDEXES.update(indexes)
        _STORES.update(stores)


def _save() -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps((_INDEXES, _STORES)))


def reset() -> None:
    global _loaded
    _INDEXES.clear()
    _STORES.clear()
    _loaded = True
    _state_path().unlink(missing_ok=True)


# ----------------------------------------------------------------------------- OData filters

TOKEN = re.compile(r"\s*(\(|\)|,|'(?:[^']|'')*'|[A-Za-z_][\w./]*|-?\d+(?:\.\d+)?(?:[-T:.\dZ+]*)|:)")


def _tokens(expr: str) -> list[str]:
    out, pos = [], 0
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"unsupported filter near: {expr[pos:pos + 20]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def _literal(tok: str):
    if tok.startswith("'"):
        return tok[1:-1].replace("''", "'")
    if tok in ("true", "false"):
        return tok == "true"
    if tok == "null":
        return None
    try:
        return float(tok) if re.fullmatch(r"-?\d+(\.\d+)?", tok) else tok
    except ValueError:
        return tok


def compile_filter(expr: str | None):
    """Subset of OData used by the labs: eq ne gt ge lt le, and/or/not, parentheses,
    search.in(field, 'a,b'[, ',']), field/any(x: x eq 'v')."""
    if not expr:
        return lambda doc: True
    toks = _tokens(expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(expected=None):
        nonlocal pos
        tok = toks[pos]
        if expected and tok.lower() != expected:
            raise ValueError(f"expected {expected}, got {tok}")
        pos += 1
        return tok

    def compare(a, op, b):
        if a is None or b is None:
            return (a == b) if op == "eq" else (a != b) if op == "ne" else False
        if isinstance(b, float) and not isinstance(a, (int, float)):
            try:
                a = float(a)
            except (TypeError, ValueError):
                return False
        a, b = (str(a), str(b)) if not isinstance(b, float) else (a, b)
        return {"eq": a == b, "ne": a != b, "gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[op]

    def primary():
        tok = peek()
        if tok == "(":
            take("(")
            node = disjunction()
            take(")")
            return node
        if tok.lower() == "not":
            take()
            inner = primary()
            return lambda d: not inner(d)
        if tok.lower() == "search.in":
            take()
            take("(")
            fld = take()
            take(",")
            values = _literal(take())
            sep = ","
            if peek() == ",":
                take(",")
                sep = _literal(take())
            take(")")
            allowed = {v.strip() for v in values.split(sep)}
            return lambda d: str(d.get(fld)) in allowed
        name = take()
        if name.endswith("/any"):
            fld = name[:-4]
            take("(")
            var = take()
            take(":")
            take(var)
            op = take().lower()
            value = _literal(take())
            take(")")
            return lambda d: any(compare(x, op, value) for x in (d.get(fld) or []))
        op = take().lower()
        value = _literal(take())
        return lambda d: compare(d.get(name), op, value)

    def conjunction():
        node = primary()
        while peek() and peek().lower() == "and":
            take()
            left, right = node, primary()
            node = lambda d, left=left, right=right: left(d) and right(d)  # noqa: E731
        return node

    def disjunction():
        node = conjunction()
        while peek() and peek().lower() == "or":
            take()
            left, right = node, conjunction()
            node = lambda d, left=left, right=right: left(d) or right(d)  # noqa: E731
        return node

    result = disjunction()
    if pos != len(toks):
        raise ValueError(f"unexpected token {toks[pos]!r} in filter")
    return result


# ------------------------------------------------------------------------------- AI Search

def _as_dict(obj) -> dict:
    return obj.as_dict() if hasattr(obj, "as_dict") else dict(obj)


@dataclass
class _Index:
    name: str
    schema: dict
    docs: dict[str, dict] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return next(f["name"] for f in self.schema["fields"] if f.get("key"))

    @property
    def vector_field(self) -> dict | None:
        return next((f for f in self.schema["fields"] if f.get("dimensions")), None)

    def compression(self) -> dict | None:
        vf = self.vector_field
        if not vf:
            return None
        vs = self.schema.get("vectorSearch") or {}
        profile = next((p for p in vs.get("profiles", []) if p.get("name") == vf.get("vectorSearchProfile")), {})
        name = profile.get("compression")
        return next((c for c in vs.get("compressions", []) if c.get("name") == name), None) if name else None


class SimulatedSearchClient:
    def __init__(self, name: str):
        self.name = name

    @property
    def _index(self) -> _Index:
        _load()
        if self.name not in _INDEXES:
            raise KeyError(f"index {self.name} does not exist")
        return _INDEXES[self.name]

    def _put(self, documents, merge: bool):
        idx, results = self._index, []
        for doc in documents:
            k = doc[idx.key]
            idx.docs[k] = {**idx.docs.get(k, {}), **doc} if merge else dict(doc)
            results.append(SimpleNamespace(key=k, succeeded=True, status_code=201))
        _save()
        return results

    def upload_documents(self, documents):
        return self._put(documents, merge=False)

    def merge_or_upload_documents(self, documents):
        return self._put(documents, merge=True)

    def delete_documents(self, documents):
        idx = self._index
        for doc in documents:
            idx.docs.pop(doc[idx.key], None)
        _save()
        return [SimpleNamespace(key=d[idx.key], succeeded=True) for d in documents]

    def get_document(self, key: str):
        return dict(self._index.docs[key])

    def get_document_count(self) -> int:
        return len(self._index.docs)

    def search(self, search_text: str | None = None, *, vector_queries=None, filter: str | None = None,
               top: int = 50, select=None, query_type=None, semantic_configuration_name=None,
               scoring_profile: str | None = None, scoring_parameters=None, order_by=None, **_):
        idx = self._index
        keep = compile_filter(filter)
        docs = [d for d in idx.docs.values() if keep(d)]
        if not docs:
            return iter([])
        text_fields = [f["name"] for f in idx.schema["fields"]
                       if f.get("searchable") and f.get("type") == "Edm.String"]
        scores: dict[str, float] = {}
        keyword_rank = vector_rank = None
        if search_text and search_text != "*":
            texts = [" ".join(str(d.get(f, "")) for f in text_fields) for d in docs]
            kw = bm25(search_text, texts)
            kw = self._apply_profile(idx, docs, kw, scoring_profile, scoring_parameters)
            keyword_rank = [i for i in np.argsort(-kw) if kw[i] > 0][:max(top, 50)]
            scores_kw = {docs[i][idx.key]: float(kw[i]) for i in keyword_rank}
        if vector_queries:
            q = vector_queries[0]
            qd = _as_dict(q)
            vec = np.asarray(qd.get("vector"), dtype=np.float32)
            k = int(qd.get("k") or qd.get("k_nearest_neighbors") or 50)
            sims = self._knn(docs, vec, k, qd)
            vector_rank = [i for i in np.argsort(-sims)][:k]
            scores_vec = {docs[i][idx.key]: float(sims[i]) for i in vector_rank}
        if keyword_rank is not None and vector_rank is not None:          # hybrid: RRF, k = 60
            for rank_list in (keyword_rank, vector_rank):
                for r, i in enumerate(rank_list, start=1):
                    key = docs[i][idx.key]
                    scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r)
        elif keyword_rank is not None:
            scores = scores_kw
        elif vector_rank is not None:
            scores = scores_vec
        else:
            scores = {d[idx.key]: 1.0 for d in docs}
        by_key = {d[idx.key]: d for d in docs}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        rerank = {}
        if query_type == "semantic" and ranked:                             # cross-encoder stand-in
            pool = ranked[:50]
            texts = [" ".join(str(by_key[k].get(f, "")) for f in text_fields) for k, _ in pool]
            kw = minmax(bm25(search_text or "", texts)) if search_text else np.zeros(len(pool))
            base = minmax(np.array([s for _, s in pool]))
            rerank = {k: float(4 * (0.5 * kw[i] + 0.5 * base[i])) for i, (k, _) in enumerate(pool)}
            ranked = sorted(pool, key=lambda kv: -rerank[kv[0]]) + ranked[50:]
        if order_by:
            fld, *direction = order_by[0].split()
            ranked.sort(key=lambda kv: str(by_key[kv[0]].get(fld) or ""),
                        reverse=bool(direction and direction[0].lower() == "desc"))
        results = []
        vector_name = (idx.vector_field or {}).get("name")
        for k, s in ranked[:top]:
            d = {f: v for f, v in by_key[k].items() if f != vector_name and (not select or f in select)}
            d["@search.score"] = s
            if k in rerank:
                d["@search.reranker_score"] = rerank[k]
            results.append(d)
        return iter(results)

    def _knn(self, docs, vec, k, qd) -> np.ndarray:
        idx = self._index
        name = idx.vector_field["name"]
        matrix = np.asarray([d.get(name) or np.zeros_like(vec) for d in docs], dtype=np.float32)
        exact = matrix @ vec / np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec), 1e-12)
        comp = idx.compression()
        if not comp:
            return exact
        kind = comp.get("kind")
        if kind == "binaryQuantization":
            approx = (np.sign(matrix) @ np.sign(vec)) / matrix.shape[1]
        else:                                              # scalarQuantization, int8
            lo, hi = matrix.min(axis=0), matrix.max(axis=0)
            scale = np.where(hi - lo < 1e-9, 1.0, (hi - lo) / 255.0)
            quantized = np.round((matrix - lo) / scale) * scale + lo
            approx = quantized @ vec / np.maximum(np.linalg.norm(quantized, axis=1) * np.linalg.norm(vec), 1e-12)
        rescoring = comp.get("rescoringOptions") or {}
        if rescoring.get("enableRescoring", False):
            oversample = float(qd.get("oversampling") or rescoring.get("defaultOversampling") or 4.0)
            pool = np.argsort(-approx)[: int(k * oversample)]
            approx = approx.copy()
            approx[pool] = exact[pool] + 1.0        # rescored candidates go first, in exact order
        return approx

    @staticmethod
    def _apply_profile(idx, docs, scores, profile_name, parameters):
        if not profile_name:
            return scores
        profile = next((p for p in idx.schema.get("scoringProfiles", []) if p.get("name") == profile_name), None)
        if not profile:
            raise ValueError(f"scoring profile {profile_name} not found")
        params = dict(p.split("-", 1) for p in (parameters or []))
        boost = np.ones(len(docs), dtype=np.float32)
        now = datetime.now(UTC)
        for fn in profile.get("functions", []):
            fld, weight = fn.get("fieldName"), float(fn.get("boost", 1))
            if fn.get("type") == "freshness":
                days = _duration_days(fn["freshness"]["boostingDuration"])
                for i, d in enumerate(docs):
                    when = _parse_time(d.get(fld))
                    if when:
                        age = max(0.0, (now - when).days)
                        boost[i] += (weight - 1) * max(0.0, 1 - age / days)
            elif fn.get("type") == "tag":
                tags = set(params.get(fn["tag"]["tagsParameter"], "").split(","))
                for i, d in enumerate(docs):
                    if tags & set(d.get(fld) or []):
                        boost[i] += weight - 1
        return scores * boost


def _duration_days(value) -> float:
    if hasattr(value, "days"):
        return max(value.days, 1)
    m = re.fullmatch(r"P(\d+)D", str(value))
    return float(m.group(1)) if m else 365.0


def _parse_time(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class SimulatedIndexClient:
    def create_or_update_index(self, index):
        _load()
        schema = _as_dict(index)
        name = schema["name"]
        existing = _INDEXES.get(name)
        _INDEXES[name] = _Index(name, schema, existing.docs if existing else {})
        _save()
        return index

    def get_index(self, name):
        _load()
        return SimpleNamespace(**_INDEXES[name].schema)

    def delete_index(self, index):
        _load()
        _INDEXES.pop(index if isinstance(index, str) else index.name, None)
        _save()

    def list_index_names(self):
        _load()
        return iter(sorted(_INDEXES))

    def get_search_client(self, name):
        return SimulatedSearchClient(name)

    def get_index_statistics(self, name) -> dict:
        _load()
        idx = _INDEXES[name]
        vf = idx.vector_field
        text_bytes = sum(len(json.dumps({k: v for k, v in d.items() if not vf or k != vf["name"]}))
                         for d in idx.docs.values())
        n, dims = len(idx.docs), (vf or {}).get("dimensions", 0)
        comp = idx.compression()
        per_vector = dims * 4
        if comp:
            per_vector = dims / 8 if comp.get("kind") == "binaryQuantization" else dims
        keep_originals = vf and (vf.get("stored", True) or bool(comp))
        return {"document_count": n, "vector_index_size": int(n * per_vector),
                "storage_size": int(text_bytes + (n * dims * 4 if keep_originals else 0) + n * per_vector),
                "simulated": True}


# --------------------------------------------------------------------- Foundry memory store

class _MemoryStores:
    """Subset of project.beta.memory_stores. Extraction is a rule-based stand-in for the model."""

    def create(self, *, name, definition=None, description=None, metadata=None, **_):
        _load()
        _STORES.setdefault(name, {"definition": _as_dict(definition) if definition else {}, "items": {}})
        _save()
        return SimpleNamespace(name=name, description=description)

    def get(self, name):
        _load()
        if name not in _STORES:
            raise KeyError(name)
        return SimpleNamespace(name=name, **_STORES[name]["definition"])

    def delete(self, name):
        _STORES.pop(name, None)

    def create_memory(self, name, *, scope, content, kind):
        item = SimpleNamespace(memory_id=uuid.uuid4().hex, scope=scope, content=content, kind=str(kind),
                               updated_at=datetime.now(UTC))
        _STORES[name]["items"][item.memory_id] = item
        _save()
        return item

    def update_memory(self, name, memory_id, *, content):
        item = _STORES[name]["items"][memory_id]
        item.content, item.updated_at = content, datetime.now(UTC)
        return item

    def delete_memory(self, name, memory_id):
        _STORES[name]["items"].pop(memory_id, None)
        return SimpleNamespace(deleted=True)

    def list_memories(self, name, *, scope, kind=None, **_):
        _load()
        return iter([i for i in _STORES[name]["items"].values()
                     if i.scope == scope and (kind is None or str(kind) == i.kind)])

    def delete_scope(self, name, *, scope):
        items = _STORES[name]["items"]
        for k in [k for k, i in items.items() if i.scope == scope]:
            items.pop(k)
        _save()
        return SimpleNamespace(deleted=True)

    def begin_update_memories(self, name, *, scope, items, update_delay=None, **_):
        """Rule-based extraction: user statements -> user_profile, the exchange -> chat_summary."""
        messages = [items] if isinstance(items, str) else list(items)
        ops = []
        user_lines = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        for line in user_lines:
            if re.match(r"(?i)\s*(i|we|my|our)\b", line):
                ops.append(self.create_memory(name, scope=scope, content=line.strip(), kind="user_profile"))
        summary = " | ".join(str(m.get("content", ""))[:120] for m in messages if isinstance(m, dict))
        if summary:
            ops.append(self.create_memory(name, scope=scope, content=summary[:600], kind="chat_summary"))
        result = SimpleNamespace(memory_operations=[SimpleNamespace(kind="create", memory_item=o) for o in ops],
                                 usage=SimpleNamespace(total_tokens=sum(len(o.content) // 4 for o in ops)))
        return SimpleNamespace(result=lambda timeout=None: result, done=lambda: True)

    def search_memories(self, name, *, scope, items=None, options=None, **_):
        _load()
        query = items if isinstance(items, str) else " ".join(
            str(m.get("content", "")) for m in (items or []) if isinstance(m, dict))
        candidates = [i for i in _STORES[name]["items"].values() if i.scope == scope]
        limit = int(_as_dict(options).get("max_memories", 5)) if options else 5
        if not candidates:
            return SimpleNamespace(search_id=uuid.uuid4().hex, memories=[])
        scores = bm25(query, [c.content for c in candidates]) if query else np.ones(len(candidates))
        order = np.argsort(-scores)[:limit]
        return SimpleNamespace(search_id=uuid.uuid4().hex,
                               memories=[SimpleNamespace(memory_item=candidates[i]) for i in order])


class _Embeddings:
    def create(self, *, model, input, **_):
        from pdfmem.embedstore import HashingModel
        vectors = HashingModel(256).encode(list(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v.tolist()) for i, v in enumerate(vectors)])


class SimulatedProject:
    def __init__(self):
        self.beta = SimpleNamespace(memory_stores=_MemoryStores())

    def get_openai_client(self):
        return SimpleNamespace(embeddings=_Embeddings())


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, (time.perf_counter() - started) * 1000
