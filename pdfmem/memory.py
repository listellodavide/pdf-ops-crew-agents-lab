"""Shared plumbing: the memory item, safe keys, document dates, the embedder of the snapshot."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pdfmem import config
from pdfmem import embedstore as es


@dataclass
class MemoryItem:
    id: str
    text: str
    document: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    kind: str = "chunk"
    created_at: str | None = None
    entities: list[str] = field(default_factory=list)
    score: float = 0.0
    attributes: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.page_start is None:
            return self.document or self.id
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"p.{self.page_start}-{self.page_end}"
        return f"{self.document} {pages}"

    @classmethod
    def from_search(cls, doc: dict, kind: str = "chunk") -> MemoryItem:
        attrs = doc.get("attributes") or "{}"
        return cls(id=doc["id"], text=doc.get("text", ""), document=doc.get("document"),
                   page_start=doc.get("page_start", doc.get("page")), page_end=doc.get("page_end", doc.get("page")),
                   kind=doc.get("kind", kind), created_at=_iso(doc.get("created_at")),
                   entities=list(doc.get("entities") or []),
                   score=float(doc.get("@search.reranker_score", doc.get("@search.score", 0.0))),
                   attributes=json.loads(attrs) if isinstance(attrs, str) else dict(attrs))


def _iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def safe_key(raw: str) -> str:
    """AI Search keys allow letters, digits, '_', '-' and '='. Base64url keeps them unique and
    reversible; long keys are hashed (limit 1024)."""
    key = base64.urlsafe_b64encode(raw.encode()).decode()
    return key if len(key) <= 1000 else hashlib.sha256(raw.encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def pdf_date_to_iso(value: str | None) -> str | None:
    """'D:20250314093000+01'00'' -> '2025-03-14T00:00:00Z'."""
    if not value:
        return None
    digits = value.removeprefix("D:")[:8]
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}T00:00:00Z"
    return None


def document_dates(ingestion_dir: Path = config.INGESTION_DIR) -> dict[str, str]:
    """Original PDF name -> creation date, from the Phase 1 *.metadata.json files."""
    dates = {}
    for meta in Path(ingestion_dir).glob("*.metadata.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            source = data["source"]
            when = pdf_date_to_iso(source.get("pdf_metadata", {}).get("creationDate"))
            if when:
                dates[source["file"]] = when
        except (KeyError, ValueError):
            continue
    return dates


_EMBEDDERS: dict[str, es.Embedder] = {}


def embedder_for(meta: dict) -> es.Embedder:
    """The model recorded in the Parquet snapshot: queries must use the same vector space."""
    name = meta["model_name"]
    if name not in _EMBEDDERS:
        spec = es.load_model_spec(meta.get("model_key", name), config.MODELS_CONFIG)
        spec = es.ModelSpec(spec.key, name, meta.get("query_prefix", ""), meta.get("passage_prefix", ""),
                            spec.model_kwargs, spec.encode_kwargs, spec.provider)
        _EMBEDDERS[name] = es.Embedder(spec)
    return _EMBEDDERS[name]
