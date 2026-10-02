#!/usr/bin/env python3
"""Phase 2a: embed text and tabular files into a local Parquet store (one file per source).

Inputs (files or folders; default: the Phase 1 output folder)
    text     .txt .md .markdown .rst .log    Phase 1 .txt files keep their page numbers
    tables   .csv .tsv .parquet .orc .jsonl .ndjson .json .arrow .feather .ipc .xlsx .xls

Tables are read in batches (--batch-rows), so a multi-million-row CSV, Parquet or ORC file never
has to fit in memory. Each row becomes one chunk (or several if its text is longer than
--chunk-size); the other columns are kept as JSON in the "attributes" column.

    python .\\embedding-builder.py
    python .\\embedding-builder.py .\\data\\parts.csv --text-column description --text-column vehicle
    python .\\embedding-builder.py .\\data\\catalog.orc .\\data\\esg.parquet --model e5small
    python .\\embedding-builder.py .\\pdf-inbox-ingestion .\\data --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
from pdfmem import embedstore as es  # noqa: E402

DEFAULT_INPUT = Path("pdf-inbox-ingestion")
DEFAULT_STORE = DEFAULT_INPUT / "embeddings"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "models.json"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2a: build the local Parquet embedding store.")
    p.add_argument("inputs", nargs="*", help=f"files or folders (default: {DEFAULT_INPUT})")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="store folder (default: %(default)s)")
    p.add_argument("--model", default="e5base",
                   help="key in configs/models.json or a sentence-transformers name/folder (default: %(default)s)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="model registry JSON")
    p.add_argument("--device", help="cpu, cuda or mps (default: automatic)")
    p.add_argument("--text-column", action="append", default=[],
                   help="table column(s) to embed; repeat to combine (default: longest text column)")
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--batch-rows", type=int, default=10000, help="table rows read per batch (default: %(default)s)")
    p.add_argument("--encode-batch", type=int, default=32, help="texts per model forward pass (default: %(default)s)")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--force", action="store_true", help="re-embed sources even if unchanged")
    return p.parse_args(argv)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows consoles
    args = parse_arguments()
    if args.chunk_overlap >= args.chunk_size:
        print("[ERROR] --chunk-overlap must be smaller than --chunk-size", file=sys.stderr)
        return 1
    store = Path(args.store)
    spec = es.load_model_spec(args.model, Path(args.config))
    inputs = [Path(p) for p in args.inputs] or [DEFAULT_INPUT]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print(f"[ERROR] not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    sources = es.discover(inputs, exclude=[store])
    if not sources:
        print("[ERROR] no supported files found", file=sys.stderr)
        return 1
    print(f"[PHASE 2a] {len(sources)} source(s), model {spec.key} ({spec.model_name}) -> {store / spec.key}")

    embedder, failed, started = None, 0, time.perf_counter()
    for path in sources:
        fmt = es.source_format(path)
        out = es.store_file(store, spec, path)
        wanted = {"schema": es.SCHEMA_VERSION, "model_key": spec.key, "model_name": spec.model_name,
                  "query_prefix": spec.query_prefix, "passage_prefix": spec.passage_prefix,
                  "normalized": True, "source": path.name, "source_format": fmt,
                  "source_sha256": es.sha256_of(path), "chunk_size": args.chunk_size,
                  "chunk_overlap": args.chunk_overlap,
                  "text_columns": args.text_column if fmt in es.TABLE_FORMATS.values() else []}
        if not args.force and es.is_current(out, wanted):
            print(f"[SKIP] {path.name}: unchanged")
            continue
        print(f"[FILE] {path.name} ({fmt})", flush=True)
        try:
            if embedder is None:      # load the model only when there is work to do
                embedder = es.Embedder(spec, args.device, args.encode_batch)
                print(f"  model on {embedder.device}, dimension {embedder.dimension}", flush=True)
            if fmt in es.TEXT_FORMATS.values():
                rows = es.text_rows(path, args.chunk_size, args.chunk_overlap)
            else:
                rows = es.table_rows(path, fmt, args.text_column, args.batch_rows, args.sheet,
                                     args.chunk_size, args.chunk_overlap)
            t0 = time.perf_counter()
            n = es.write_source(path, out, rows, embedder, wanted)
            if n:
                size = out.stat().st_size / 1024
                print(f"  [OK] {n:,} chunks -> {out.name} ({size:,.0f} KB, {time.perf_counter() - t0:.1f}s)")
            else:
                print("  [WARN] no text to embed")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {path.name}: {e}")
    print(f"\n[done in {time.perf_counter() - started:.1f}s, {failed} failed]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
