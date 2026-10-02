"""Lab 1 / Block 2: semantic memory recall on Azure AI Search, five ways.

    keyword   BM25 only                                (search_text)
    vector    kNN on the compressed vectors            (vector_queries)
    hybrid    both, fused by Reciprocal Rank Fusion    (search_text + vector_queries)
    semantic  hybrid, then the semantic ranker         (+ query_type="semantic")
    fresh     hybrid with the scoring profile          (+ scoring_profile, entity tags):
              newer documents and matching entities rank higher (multi-dimensional recall)

Filters (document names, dates) run inside the service before ranking.

Run:
    python -m pdfmem.semantic "What torque for the caliper bolts?" --mode hybrid
    python -m pdfmem.evaluate --modes keyword vector hybrid semantic fresh
"""

from __future__ import annotations

import argparse
import os

from azure.search.documents.models import VectorizedQuery

from pdfmem import cloud, config, schema
from pdfmem.memory import MemoryItem, embedder_for
from pdfmem.optimize import load_snapshot
from pdfmem.text import entities

MODES = ("keyword", "vector", "hybrid", "semantic", "fresh")


def documents_filter(names: list[str]) -> str | None:
    """OData filter: document is one of `names`. Uses search.in, which is fast on long lists."""
    if not names:
        return None
    # TODO 1.2-b: return an OData filter string with search.in on the field "document",
    # using "|" as the delimiter (document names can contain commas).
    raise NotImplementedError("TODO 1.2-b: see the comment above")


def newer_than(iso_date: str) -> str:
    return f"created_at ge {iso_date}"


def search_kwargs(mode: str, query: str, vector: list[float] | None, k: int,
                  filter: str | None = None) -> dict:
    """Keyword arguments for SearchClient.search() for one recall mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    # TODO 1.2-a: build the arguments. Every mode gets top=k, filter=filter and
    # select=["id", "text", "document", "page_start", "page_end", "created_at", "entities", "attributes"].
    #   keyword  -> search_text=query
    #   vector   -> search_text=None, vector_queries=[VectorizedQuery(vector=..., k_nearest_neighbors=k, fields="embedding")]
    #   hybrid   -> both of the above
    #   semantic -> hybrid + query_type="semantic", semantic_configuration_name=schema.SEMANTIC_CONFIG
    #   fresh    -> hybrid + scoring_profile=schema.SCORING_PROFILE and, when the query has entities,
    #               scoring_parameters=[f"{schema.ENTITY_TAGS_PARAMETER}-{','.join(sorted(entities(query)))}"]
    raise NotImplementedError("TODO 1.2-a: see the comment above")
    return kwargs


class SemanticMemory:
    """Recall over the team's chunks index. Queries are embedded with the snapshot's model."""

    def __init__(self, variant: str | None = None, meta: dict | None = None):
        self.variant = variant or os.environ.get("PDFMEM_CHUNKS_VARIANT", "scalar")
        self.index = cloud.index_name("chunks", self.variant)
        self.client = cloud.search_client(self.index)
        self.meta = meta if meta is not None else load_snapshot()[1]
        self._embedder = None

    def ensure_memories_index(self) -> str:
        """Create (or update) the team's memories index with the snapshot's vector size."""
        name = cloud.index_name("memories")
        cloud.index_client().create_or_update_index(schema.memories_index(name, int(self.meta["dimension"])))
        return name

    def embed(self, text: str) -> list[float]:
        if self._embedder is None:
            self._embedder = embedder_for(self.meta)
        return [float(x) for x in self._embedder.query(text)]

    def recall(self, query: str, k: int = 5, mode: str = "hybrid", documents: list[str] | None = None,
               filter: str | None = None) -> list[MemoryItem]:
        vector = None if mode == "keyword" else self.embed(query)
        clauses = [c for c in (documents_filter(documents or []), filter) if c]
        combined = " and ".join(f"({c})" for c in clauses) if clauses else None
        results = self.client.search(**search_kwargs(mode, query, vector, k, combined))
        return [MemoryItem.from_search(r) for r in results]


def main() -> int:
    p = argparse.ArgumentParser(description="Recall from the resident semantic memory.")
    p.add_argument("query")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--document", action="append", default=[])
    args = p.parse_args()
    memory = SemanticMemory()
    for n, item in enumerate(memory.recall(args.query, args.k, args.mode, args.document), start=1):
        print(f"{n}. {item.score:.3f}  {item.citation}\n   {item.text[:200]!r}")
    return 0


if __name__ == "__main__":
    _ = config  # .env loaded through pdfmem.config
    raise SystemExit(main())
