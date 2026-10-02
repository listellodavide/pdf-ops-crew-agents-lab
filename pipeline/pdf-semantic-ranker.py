#!/usr/bin/env python3
"""Phase 2b: rank documents (or single chunks/rows) for a topic, from the local Parquet store.

Run embedding-builder.py first. The query is embedded with the model recorded in the store, then
compared with every stored chunk by cosine similarity:

    cos(q, d) = (q . d) / (||q|| * ||d||)

    python .\\pdf-semantic-ranker.py "Sustainability"
    python .\\pdf-semantic-ranker.py "reduction of carbon emissions" --top 5 --json ranking.json
    python .\\pdf-semantic-ranker.py "cylinder head gasket" --level chunk --format csv --format orc
    python .\\pdf-semantic-ranker.py "brake pads" --where "source_format = 'csv' AND n_chars > 40"
    python .\\pdf-semantic-ranker.py "tire" --method keyword          # see why keywords are noisy

Methods
    semantic  cosine similarity (default)
    keyword   BM25 on the stored chunk text, no model needed
    hybrid    Reciprocal Rank Fusion of both rankings (k = 60)

Levels
    document  group chunks by document (default). Score, after rescaling chunk scores to 0..1:
              0.6 * best chunk + 0.3 * mean of 3 best + 0.1 * coverage, where coverage is the
              number of the document's chunks in the corpus top 50, divided by 5 (max 1).
    chunk     the top chunks or table rows themselves, best first (argpartition top-k)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
from pdfmem import embedstore as es  # noqa: E402

DEFAULT_STORE = Path("pdf-inbox-ingestion") / "embeddings"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "models.json"
RRF_K = 60
TOP_K_MEAN = 3
COVERAGE_POOL = 50
COVERAGE_FULL = 5
WEIGHTS = (0.6, 0.3, 0.1)
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


def bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    avg = sum(map(len, docs)) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    scores = np.zeros(n, dtype=np.float32)
    for term in set(tokenize(query)):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
    return scores


def rrf(*score_lists: np.ndarray, k: int = RRF_K) -> np.ndarray:
    fused = np.zeros(len(score_lists[0]), dtype=np.float32)
    for scores in score_lists:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        fused += np.where(scores > 0, 1.0 / (k + ranks), 0.0).astype(np.float32)
    return fused


def rescale(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    return np.zeros_like(scores) if high - low < 1e-9 else (scores - low) / (high - low)


def rank_documents(frame: pl.DataFrame, raw: np.ndarray, show: int) -> list[dict]:
    scaled = rescale(raw)
    pool = np.zeros(len(raw), dtype=bool)
    pool[es.top_k(raw, COVERAGE_POOL)] = True
    scored = frame.select("document", "source_format", "page_start", "page_end", "record", "text").with_columns(
        pl.Series("raw", raw), pl.Series("scaled", scaled), pl.Series("in_pool", pool & (raw > 0)))
    ranked = []
    for (document,), group in scored.group_by("document", maintain_order=True):
        group = group.sort("scaled", descending=True)
        best = float(group["scaled"][0])
        top_mean = float(group["scaled"].head(TOP_K_MEAN).mean())
        relevant = int(group["in_pool"].sum())
        coverage = min(1.0, relevant / COVERAGE_FULL)
        score = WEIGHTS[0] * best + WEIGHTS[1] * top_mean + WEIGHTS[2] * coverage
        ranked.append({"document": document, "score": round(score, 4), "best": round(best, 4),
                       "top_mean": round(top_mean, 4), "coverage": round(coverage, 2),
                       "relevant_chunks": relevant, "total_chunks": group.height,
                       "best_chunks": group.head(show).drop("in_pool").to_dicts()})
    return sorted(ranked, key=lambda d: -d["score"])


def where_label(row: dict) -> str:
    if row.get("page_start") is not None:
        ps, pe = row["page_start"], row["page_end"]
        return f"p.{ps}" if ps == pe else f"p.{ps}-{pe}"
    return f"row {row['record']}" if row.get("source_format", "") not in ("text", "pdf-text") else "text"


def snippet(text: str, width: int = 200) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2b: rank stored documents or rows for a query.")
    p.add_argument("query", help='topic or question, e.g. "Sustainability"')
    p.add_argument("--store", default=str(DEFAULT_STORE), help="store folder (default: %(default)s)")
    p.add_argument("--model", default="e5base", help="model key used when building (default: %(default)s)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--device", help="cpu, cuda or mps (default: automatic)")
    p.add_argument("--method", choices=("semantic", "keyword", "hybrid"), default="semantic")
    p.add_argument("--level", choices=("document", "chunk"), default="document")
    p.add_argument("--include", action="append", default=[], help="document name pattern, e.g. \"*catalog*\"")
    p.add_argument("--format", dest="formats", action="append", default=[],
                   help="source format filter: pdf-text, text, csv, tsv, parquet, orc, ndjson, json, ipc, excel")
    p.add_argument("--where", help="SQL condition on store columns, e.g. \"page_start <= 10\"")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--chunks", type=int, default=2, help="best chunks shown per document (default: %(default)s)")
    p.add_argument("--json", dest="json_out", help="write the full result to this JSON file")
    return p.parse_args(argv)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_arguments()
    started = time.perf_counter()
    spec = es.load_model_spec(args.model, Path(args.config))
    try:
        lazy, meta = es.open_store(Path(args.store), spec.key)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    try:
        frame = es.apply_filters(lazy, args.include, args.formats, args.where).collect()
    except Exception as e:
        print(f"[ERROR] filter failed: {str(e).splitlines()[0]}", file=sys.stderr)
        return 1
    if frame.is_empty():
        print("[ERROR] no chunks left after filtering", file=sys.stderr)
        return 1
    print(f"[PHASE 2b] {frame['document'].n_unique()} documents, {frame.height:,} chunks, "
          f"model {meta.get('model_name')}, method {args.method}", flush=True)

    semantic = keyword = None
    if args.method in ("semantic", "hybrid"):
        # the query must use the same model and prefix as the stored passages
        stored = es.ModelSpec(spec.key, meta["model_name"], meta.get("query_prefix", ""),
                              meta.get("passage_prefix", ""), spec.model_kwargs, spec.encode_kwargs)
        embedder = es.Embedder(stored, args.device)
        matrix = frame["embedding"].to_numpy()                 # Array(Float32, dim) -> 2-D float32
        semantic = es.cosine_similarity(embedder.query(args.query), matrix)
    if args.method in ("keyword", "hybrid"):
        keyword = bm25_scores(args.query, frame["text"].to_list())
    raw = {"semantic": semantic, "keyword": keyword}.get(args.method)
    if args.method == "hybrid":
        raw = rrf(semantic, keyword)
    label = {"semantic": "cos", "keyword": "bm25", "hybrid": "rrf"}[args.method]

    print(f'\nQuery: "{args.query}"\n')
    result: dict = {"query": args.query, "method": args.method, "level": args.level,
                    "model": meta.get("model_name")}
    if args.level == "chunk":
        idx = es.top_k(raw, args.top)
        hits = frame[idx].drop("embedding").with_columns(pl.Series("score", raw[idx])).to_dicts()
        for n, h in enumerate(hits, start=1):
            print(f"{n:>3}  {label} {h['score']:.3f}  {h['document']}  {where_label(h)}")
            print(f"       {snippet(h['text'])}")
            if h["attributes"] not in ("{}", None):
                print(f"       attributes: {snippet(h['attributes'], 160)}")
        result["chunks"] = hits
    else:
        ranking = rank_documents(frame.drop("embedding"), raw, args.chunks)
        print(f"{'#':>3}  {'score':>6}  {'best':>5}  {'top3':>5}  {'hits':>7}  document")
        for n, d in enumerate(ranking[: args.top], start=1):
            print(f"{n:>3}  {d['score']:6.3f}  {d['best']:5.2f}  {d['top_mean']:5.2f}  "
                  f"{d['relevant_chunks']:>3}/{d['total_chunks']:<3}  {d['document']}")
            for c in d["best_chunks"]:
                print(f"       {where_label(c):<9} {label} {c['raw']:.3f}  {snippet(c['text'])}")
        result["documents"] = ranking
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                                       encoding="utf-8")
        print(f"\n[OK] written to {args.json_out}")
    print(f"\n[done in {time.perf_counter() - started:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
