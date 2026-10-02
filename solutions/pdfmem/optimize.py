"""Lab 1 / Block 1: from the Parquet snapshot to resident semantic memory in Azure AI Search.

The Parquet files are the intermediate memory: every chunk Phase 2a embedded, including running
headers, repeated legal notices and fragments. Before they become resident memory we:

    1. drop chunks too short to answer anything
    2. drop exact duplicates (same normalized text)
    3. drop near-duplicates (cosine >= threshold to a chunk already kept)       TODO 1.1-a
    4. upload to an index whose vectors are compressed (none / scalar / binary)  TODO 1.1-b
       with documents carrying date and entities for later filtering            TODO 1.1-c

Run:
    python -m pdfmem.optimize --variants none scalar binary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from pdfmem import cloud, config, schema
from pdfmem import embedstore as es
from pdfmem.memory import document_dates, safe_key
from pdfmem.text import entities


@dataclass
class OptimizeReport:
    input_chunks: int
    dropped_short: int
    dropped_exact: int
    dropped_near: int
    kept: int

    @property
    def reduction(self) -> float:
        return 1 - self.kept / max(self.input_chunks, 1)


def load_snapshot(store_dir: Path = config.STORE_DIR, model: str = config.MODEL) -> tuple[pl.DataFrame, dict]:
    spec = es.load_model_spec(model, config.MODELS_CONFIG)
    lazy, meta = es.open_store(Path(store_dir), spec.key)
    return lazy.collect(), meta


def drop_short(frame: pl.DataFrame, min_chars: int = 60) -> pl.DataFrame:
    return frame.filter(pl.col("n_chars") >= min_chars)


def drop_exact_duplicates(frame: pl.DataFrame) -> pl.DataFrame:
    """Same text after lowercasing and collapsing whitespace: keep the first occurrence."""
    norm = frame["text"].map_elements(lambda t: hashlib.sha1(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest(),
                                      return_dtype=pl.String)
    return frame.with_columns(norm.alias("__h")).unique(subset="__h", keep="first", maintain_order=True).drop("__h")


def drop_near_duplicates(frame: pl.DataFrame, threshold: float = 0.97) -> pl.DataFrame:
    """Greedy de-duplication on the stored unit vectors.

    Walk the chunks in order; keep a chunk only if its cosine similarity to every chunk already
    kept is below `threshold`. Vectors in the snapshot are L2-normalized, so cosine = dot product.
    Returns the frame restricted to the kept rows, in the original order.
    """
    if frame.is_empty():
        return frame
    vectors = frame["embedding"].to_numpy()
    # TODO 1.1-a: compute the list `keep` of row indices to keep.
    # Hint: start with keep = [0]; for row i, compare vectors[i] with vectors[keep] in one
    #       matrix product and keep i if the maximum similarity is below the threshold.
    # >>> solution
    keep = [0]
    for i in range(1, len(vectors)):
        if float(np.max(vectors[keep] @ vectors[i])) < threshold:
            keep.append(i)
    # <<< solution
    return frame[keep]


def optimize(frame: pl.DataFrame, min_chars: int = 60, threshold: float = 0.97) -> tuple[pl.DataFrame, OptimizeReport]:
    n0 = frame.height
    a = drop_short(frame, min_chars)
    b = drop_exact_duplicates(a)
    c = drop_near_duplicates(b, threshold)
    return c, OptimizeReport(n0, n0 - a.height, a.height - b.height, b.height - c.height, c.height)


def to_documents(frame: pl.DataFrame, dates: dict[str, str]) -> list[dict]:
    """Rows of the snapshot -> documents for the chunks index (see schema.chunk_index)."""
    docs = []
    for row in frame.iter_rows(named=True):
        # TODO 1.1-c: build the search document for one chunk. Required fields:
        #   id (safe_key of the snapshot id), text, document, source_format, page_start, page_end,
        #   created_at (dates.get(document), may be None), entities (sorted list from text.entities),
        #   attributes (the snapshot's JSON string), embedding (list of floats).
        # >>> solution
        docs.append({
            "id": safe_key(row["id"]), "text": row["text"], "document": row["document"],
            "source_format": row["source_format"], "page_start": row["page_start"], "page_end": row["page_end"],
            "created_at": dates.get(row["document"]), "entities": sorted(entities(row["text"])),
            "attributes": row["attributes"], "embedding": [float(x) for x in row["embedding"]],
        })
        # <<< solution
    return docs


def create_chunk_index(dimensions: int, variant: str) -> str:
    # TODO 1.1-b: open pdfmem/schema.py and read compression_for(); then create (or update) the
    # index for this variant with cloud.index_client().create_or_update_index(...) and return its name.
    # >>> solution
    name = cloud.index_name("chunks", variant)
    cloud.index_client().create_or_update_index(schema.chunk_index(name, dimensions, variant))
    # <<< solution
    return name


def upload(name: str, documents: list[dict], batch: int = 500) -> int:
    client = cloud.search_client(name)
    sent = 0
    for i in range(0, len(documents), batch):
        results = client.merge_or_upload_documents(documents[i:i + batch])
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"{len(failed)} documents failed, e.g. {failed[:3]}")
        sent += len(documents[i:i + batch])
    return sent


def index_size(name: str, expected: int, wait_seconds: int = 60) -> dict:
    """Statistics lag a few seconds behind uploads on the real service: poll until counted."""
    deadline = time.time() + wait_seconds
    while True:
        stats = cloud.index_client().get_index_statistics(name)
        stats = stats if isinstance(stats, dict) else stats.as_dict()
        if stats.get("document_count", 0) >= expected or time.time() > deadline:
            return stats
        time.sleep(5)


def main() -> int:
    p = argparse.ArgumentParser(description="Lab 1 Block 1: optimize the Parquet snapshot and load it.")
    p.add_argument("--store", default=str(config.STORE_DIR))
    p.add_argument("--model", default=config.MODEL)
    p.add_argument("--ingestion", default=str(config.INGESTION_DIR), help="Phase 1 folder (document dates)")
    p.add_argument("--variants", nargs="+", default=["scalar"], choices=schema.VARIANTS)
    p.add_argument("--min-chars", type=int, default=60)
    p.add_argument("--threshold", type=float, default=0.97)
    args = p.parse_args()

    frame, meta = load_snapshot(Path(args.store), args.model)
    kept, report = optimize(frame, args.min_chars, args.threshold)
    print(f"[optimize] {report.input_chunks} chunks -> {report.kept} kept "
          f"({report.reduction:.0%} smaller): short {report.dropped_short}, exact {report.dropped_exact}, "
          f"near {report.dropped_near}")
    documents = to_documents(kept, document_dates(Path(args.ingestion)))
    rows = []
    for variant in args.variants:
        name = create_chunk_index(int(meta["dimension"]), variant)
        upload(name, documents)
        stats = index_size(name, len(documents))
        rows.append({"variant": variant, "index": name, **{k: stats.get(k) for k in
                     ("document_count", "storage_size", "vector_index_size")}})
        print(f"[load] {name}: {stats.get('document_count')} docs, storage {stats.get('storage_size', 0) / 1024:,.0f} KB, "
              f"vector index {stats.get('vector_index_size', 0) / 1024:,.0f} KB")
    out = config.MEMORY_DIR / "lab1_block1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": asdict(report), "indexes": rows}, indent=2), encoding="utf-8")
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
