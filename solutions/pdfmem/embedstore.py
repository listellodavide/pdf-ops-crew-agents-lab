"""Local embedding store: Apache Parquet files read and filtered with Polars.

Shared by embedding-builder.py (writes) and pdf-semantic-ranker.py (reads). Adapted from
brian-ogrady/local-embed-gen (MIT): batched reading, a JSON model registry and a zstd
ParquetWriter. Following Max Woolf's "embeddings in Parquet with Polars", vectors are stored as
a fixed-size Array(Float32, dim) column, so Polars hands back a 2-D NumPy matrix directly.

Store layout
    <store>/<model key>/<source file name>.parquet     one file per source, per model

Row schema (one row per chunk)
    id            "<source>#<record>.<chunk>"
    document      what results are grouped by: the original PDF for Phase 1 text, else the file
    source        file that was read;   source_format   pdf-text, text, csv, tsv, parquet, orc, ...
    record        row number in a table, 0 for a text document;   chunk   chunk number in it
    page_start, page_end   pages for Phase 1 text, null otherwise
    text, n_chars
    attributes    the other columns of a table row as a JSON object string ("{}" for text)
    embedding     Array(Float32, dim), L2-normalized

File-level metadata (key "embedstore") records model, dimension, prefixes, chunk settings and
the source SHA-256, so an unchanged source is not embedded twice.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "embedstore-v1"
METADATA_KEY = b"embedstore"
PAGE_MARKER = re.compile(r"^\[\[page (\d+)\]\]$", re.MULTILINE)

TEXT_FORMATS = {".txt": "text", ".md": "text", ".markdown": "text", ".text": "text", ".rst": "text",
                ".log": "text"}
TABLE_FORMATS = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".parquet": "parquet", ".orc": "orc",
                 ".jsonl": "ndjson", ".ndjson": "ndjson", ".json": "json", ".arrow": "ipc",
                 ".feather": "ipc", ".ipc": "ipc", ".xlsx": "excel", ".xlsm": "excel", ".xls": "excel"}
SKIP_SUFFIXES = (".layout.json", ".metadata.json")     # Phase 1 side files, not content


# ------------------------------------------------------------------------------------- models

@dataclass
class ModelSpec:
    key: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    model_kwargs: dict | None = None
    encode_kwargs: dict | None = None
    provider: str = "sentence-transformers"     # or "foundry" (cloud) or "hashing" (offline tests)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def load_model_spec(name: str, config_path: Path) -> ModelSpec:
    """A key from configs/models.json, or any sentence-transformers name / local folder."""
    configs = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if name in configs:
        c = configs[name]
        provider = c.get("provider", "hashing" if c["model_name"].startswith("hashing:") else "sentence-transformers")
        return ModelSpec(name, c["model_name"], c.get("query_prefix", ""), c.get("passage_prefix", ""),
                         c.get("model_kwargs"), c.get("encode_kwargs"), provider)
    if name.startswith("hashing:"):
        return ModelSpec(name.replace(":", ""), name, provider="hashing")
    is_e5 = "e5" in name.lower()
    return ModelSpec(slug(Path(name).name if Path(name).exists() else name), name,
                     "query: " if is_e5 else "", "passage: " if is_e5 else "")


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HashingModel:
    """Offline stand-in for an embedding model: feature hashing of words and word bigrams.

    No download, deterministic across machines (crc32), fast. It captures shared vocabulary, not
    meaning: "carbon emissions" and "CO2 output" are far apart. Use it to run the labs offline and
    to show what a real semantic model adds.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts: list[str], **_) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[^\W_]+", text.lower())
            for token in words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]:
                h = zlib.crc32(token.encode())
                out[row, h % self.dimension] += 1.0 if (h >> 31) & 1 else -1.0
        return np.sign(out) * np.log1p(np.abs(out))


class FoundryEmbeddingModel:
    """Embeddings from a Microsoft Foundry deployment (Entra ID). The same deployment can be set as
    the Azure AI Search vectorizer, so queries are embedded in the cloud, not on the laptop."""

    def __init__(self, deployment: str, batch_size: int = 64):
        from pdfmem import cloud
        self.client = cloud.project().get_openai_client()
        self.deployment = os.environ.get("EMBEDDING_DEPLOYMENT_NAME", deployment)
        self.batch_size = batch_size
        self.dimension = len(self.encode(["dimension probe"])[0])

    def encode(self, texts: list[str], **_) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(model=self.deployment, input=texts[i:i + self.batch_size])
            out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return np.asarray(out, dtype=np.float32)


class Embedder:
    def __init__(self, spec: ModelSpec, device: str | None = None, batch_size: int = 32):
        self.spec = spec
        self.batch_size = batch_size
        if spec.provider == "foundry":
            self.device = "foundry"
            self.model = FoundryEmbeddingModel(spec.model_name)
            self.dimension = self.model.dimension
            return
        if spec.provider == "hashing" or spec.model_name.startswith("hashing:"):
            self.device = "cpu"
            self.model = HashingModel(int(spec.model_name.split(":", 1)[1]))
            self.dimension = self.model.dimension
            return
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed
        self.device = pick_device(device)
        self.model = SentenceTransformer(spec.model_name, device=self.device, **(spec.model_kwargs or {}))
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def _encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(self.model, (HashingModel, FoundryEmbeddingModel)):
            return l2_normalize(self.model.encode(texts))
        vectors = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                    show_progress_bar=False, **(self.spec.encode_kwargs or {}))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.passage_prefix + t for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text])[0]


# ----------------------------------------------------------------------------- vector maths

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """cos(q, m_i) = (q . m_i) / (||q|| * ||m_i||) for every row m_i of the matrix.

    Stored vectors are already unit length, so the denominator is 1 and this equals the dot
    product; it is computed in full anyway, so vectors from any source give a correct cosine.
    """
    query = np.asarray(query, dtype=np.float32)
    dots = matrix @ query
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return dots / np.maximum(norms, 1e-12)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first, without sorting the whole array."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(-scores[idx], kind="stable")]


# ------------------------------------------------------------------------------- chunking

def split_long(paragraph: str, size: int) -> list[str]:
    """Split an oversized paragraph on sentence ends, then on lines, then hard."""
    if len(paragraph) <= size:
        return [paragraph]
    out, buf = [], ""
    for piece in re.split(r"(?<=[.!?;])\s+|\n", paragraph):
        while len(piece) > size:
            out.append(piece[:size])
            piece = piece[size:]
        if buf and len(buf) + len(piece) + 1 > size:
            out.append(buf)
            buf = piece
        else:
            buf = f"{buf} {piece}".strip()
    if buf:
        out.append(buf)
    return out


def split_pages(text: str) -> list[tuple[int | None, str]]:
    marks = list(PAGE_MARKER.finditer(text))
    if not marks:
        return [(None, text)]
    pages = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.append((int(m.group(1)), text[m.end():end].strip()))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[int | None, int | None, str]]:
    """Pack paragraphs into chunks of about `size` characters; carry up to `overlap` characters of
    trailing paragraphs into the next chunk. Returns (page_start, page_end, text)."""
    units: list[tuple[int | None, str]] = []
    for page, body in split_pages(text):
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if para:
                units.extend((page, piece) for piece in split_long(para, size))
    chunks, current = [], []

    def flush() -> None:
        if current:
            chunks.append((current[0][0], current[-1][0], "\n\n".join(t for _, t in current)))

    for unit in units:
        if current and sum(len(t) + 2 for _, t in current) + len(unit[1]) > size:
            flush()
            carry = []
            for u in reversed(current):
                if sum(len(t) for _, t in carry) + len(u[1]) > overlap:
                    break
                carry.insert(0, u)
            current = carry
        current.append(unit)
    flush()
    return chunks


# -------------------------------------------------------------------------------- readers

@dataclass
class Row:
    document: str
    source: str
    source_format: str
    record: int
    chunk: int
    page_start: int | None
    page_end: int | None
    text: str
    attributes: str


def source_format(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(SKIP_SUFFIXES):
        return None
    suffix = path.suffix.lower()
    return TEXT_FORMATS.get(suffix) or TABLE_FORMATS.get(suffix)


def discover(inputs: list[Path], exclude: list[Path]) -> list[Path]:
    excluded = [e.resolve() for e in exclude]
    found: list[Path] = []
    for p in inputs:
        candidates = sorted(q for q in p.iterdir() if q.is_file()) if p.is_dir() else [p]
        for q in candidates:
            r = q.resolve()
            if any(r == e or e in r.parents for e in excluded):
                continue
            if source_format(q) and r not in {f.resolve() for f in found}:
                found.append(q)
    return found


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):        # Windows tools still write cp1252
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def text_rows(path: Path, size: int, overlap: int) -> Iterator[list[Row]]:
    text = read_text_file(path)
    document, fmt = path.name, "text"
    meta = path.with_name(f"{path.stem}.metadata.json")
    if PAGE_MARKER.search(text):
        fmt = "pdf-text"
        if meta.exists():
            try:
                document = json.loads(meta.read_text(encoding="utf-8"))["source"]["file"]
            except Exception:
                document = f"{path.stem}.pdf"
    rows = [Row(document, path.name, fmt, 0, i, ps, pe, t, "{}")
            for i, (ps, pe, t) in enumerate(chunk_text(text, size, overlap))]
    if rows:
        yield rows


def table_batches(path: Path, fmt: str, batch_rows: int, sheet: str | None) -> Iterator[pl.DataFrame]:
    """Yield DataFrames of at most batch_rows rows without loading big files at once."""
    if fmt == "orc":
        try:
            import pyarrow.orc as orc
        except ImportError as e:
            raise RuntimeError("this pyarrow build cannot read ORC; install pyarrow>=16") from e
        reader = orc.ORCFile(str(path))
        for stripe in range(reader.nstripes):
            frame = pl.from_arrow(reader.read_stripe(stripe))
            yield from (frame.slice(o, batch_rows) for o in range(0, frame.height, batch_rows))
        return
    if fmt in ("json", "excel"):            # no streaming reader: whole file, then slices
        if fmt == "json":
            frame = pl.read_json(path)
        else:
            frame = pl.read_excel(path, sheet_name=sheet) if sheet else pl.read_excel(path)
        yield from (frame.slice(o, batch_rows) for o in range(0, frame.height, batch_rows))
        return
    scans = {
        "csv": lambda: pl.scan_csv(path, infer_schema_length=10000, encoding="utf8-lossy"),
        "tsv": lambda: pl.scan_csv(path, separator="\t", infer_schema_length=10000, encoding="utf8-lossy"),
        "parquet": lambda: pl.scan_parquet(path),
        "ndjson": lambda: pl.scan_ndjson(path, infer_schema_length=10000),
        "ipc": lambda: pl.scan_ipc(path),
    }
    lazy = scans[fmt]()
    try:
        yield from (b for b in lazy.collect_batches(chunk_size=batch_rows) if b.height)
    except AttributeError:                   # older Polars without collect_batches
        frame = lazy.collect()
        yield from (frame.slice(o, batch_rows) for o in range(0, frame.height, batch_rows))


def choose_text_columns(frame: pl.DataFrame, requested: list[str]) -> list[str]:
    found = [c for c in requested if c in frame.columns]
    if found:
        if len(found) < len(requested):
            print(f"  [WARN] column(s) not in this file: {sorted(set(requested) - set(found))}")
        return found
    if requested:
        print(f"  [WARN] none of {requested} in this file; choosing the longest text column")
    strings = [c for c, t in frame.schema.items() if t == pl.String]
    if not strings:
        raise ValueError(f"no text column found; pass --text-column (columns: {frame.columns})")
    lengths = frame.select(pl.col(strings).str.len_chars().mean()).row(0)
    return [max(zip(lengths, strings, strict=True), key=lambda x: (x[0] or 0))[1]]


def table_rows(path: Path, fmt: str, text_columns: list[str], batch_rows: int, sheet: str | None,
               size: int, overlap: int) -> Iterator[list[Row]]:
    offset, columns = 0, None
    for frame in table_batches(path, fmt, batch_rows, sheet):
        if columns is None:
            columns = choose_text_columns(frame, text_columns)
            print(f"  text column(s): {', '.join(columns)}", flush=True)
        if len(columns) == 1:
            text_expr = pl.col(columns[0]).cast(pl.String)
        else:   # "description: ...\nvehicle: ..." keeps the field names visible to the model
            text_expr = pl.concat_str([pl.format(f"{c}: {{}}", pl.col(c).cast(pl.String)) for c in columns],
                                      separator="\n", ignore_nulls=True)
        others = [c for c in frame.columns if c not in columns]
        attrs = pl.struct(others).struct.json_encode() if others else pl.lit("{}")
        view = frame.with_row_index("__row", offset=offset).select(
            pl.col("__row"), text_expr.alias("__text"), attrs.alias("__attrs"))
        offset += frame.height
        rows: list[Row] = []
        for record, text, attributes in view.iter_rows():
            text = (text or "").strip()
            if not text:
                continue
            pieces = split_long(text, size) if len(text) > size else [text]
            rows.extend(Row(path.name, path.name, fmt, int(record), i, None, None, piece, attributes)
                        for i, piece in enumerate(pieces))
        if rows:
            yield rows


# ------------------------------------------------------------------------------ writing

def arrow_schema(dimension: int, metadata: dict) -> pa.Schema:
    return pa.schema([
        ("id", pa.string()), ("document", pa.string()), ("source", pa.string()),
        ("source_format", pa.string()), ("record", pa.int64()), ("chunk", pa.int32()),
        ("page_start", pa.int32()), ("page_end", pa.int32()), ("text", pa.string()),
        ("n_chars", pa.int32()), ("attributes", pa.string()),
        ("embedding", pa.list_(pa.float32(), dimension)),
    ], metadata={METADATA_KEY: json.dumps(metadata).encode()})


def rows_to_table(rows: list[Row], vectors: np.ndarray, schema: pa.Schema) -> pa.Table:
    dim = vectors.shape[1]
    columns = {
        "id": [f"{r.source}#{r.record}.{r.chunk}" for r in rows],
        "document": [r.document for r in rows], "source": [r.source for r in rows],
        "source_format": [r.source_format for r in rows], "record": [r.record for r in rows],
        "chunk": [r.chunk for r in rows], "page_start": [r.page_start for r in rows],
        "page_end": [r.page_end for r in rows], "text": [r.text for r in rows],
        "n_chars": [len(r.text) for r in rows], "attributes": [r.attributes for r in rows],
    }
    arrays = [pa.array(columns[f.name], type=f.type) for f in schema if f.name != "embedding"]
    arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), pa.float32()), dim))
    return pa.Table.from_arrays(arrays, schema=schema)


def store_file(store: Path, spec: ModelSpec, source: Path) -> Path:
    return store / spec.key / f"{source.name}.parquet"


def read_store_metadata(path: Path) -> dict | None:
    try:
        meta = pq.read_schema(path).metadata or {}
        return json.loads(meta[METADATA_KEY])
    except Exception:
        return None


def is_current(path: Path, wanted: dict) -> bool:
    found = read_store_metadata(path) if path.exists() else None
    keys = ("schema", "model_name", "source_sha256", "chunk_size", "chunk_overlap", "text_columns",
            "passage_prefix")
    return bool(found) and all(found.get(k) == wanted.get(k) for k in keys)


def write_source(path: Path, out: Path, rows_iter: Iterator[list[Row]], embedder: Embedder,
                 metadata: dict) -> int:
    """Embed batch by batch and stream into Parquet (zstd). Written to a temp file first, so an
    interrupted run never leaves a half file that looks complete."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    metadata = {**metadata, "dimension": embedder.dimension,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    schema = arrow_schema(embedder.dimension, metadata)
    written = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3) as writer:
        for rows in rows_iter:
            vectors = embedder.passages([r.text for r in rows])
            writer.write_table(rows_to_table(rows, vectors, schema))
            written += len(rows)
            print(f"  {written:,} chunks embedded", flush=True)
    if written:
        os.replace(tmp, out)
    else:
        tmp.unlink(missing_ok=True)
    return written


# -------------------------------------------------------------------------------- reading

def open_store(store: Path, model_key: str) -> tuple[pl.LazyFrame, dict]:
    folder = store / model_key
    files = sorted(folder.glob("*.parquet")) if folder.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no embeddings in {folder}; run embedding-builder.py first")
    metas = [read_store_metadata(f) or {} for f in files]
    models = {m.get("model_name") for m in metas}
    dims = {m.get("dimension") for m in metas}
    if len(models) > 1 or len(dims) > 1:
        raise ValueError(f"{folder} mixes models {models} or dimensions {dims}; rebuild with --force")
    return pl.scan_parquet([str(f) for f in files]), metas[0]


def apply_filters(lazy: pl.LazyFrame, include: list[str], formats: list[str], where: str | None) -> pl.LazyFrame:
    """Metadata filters run before any similarity is computed (fewer vectors to compare)."""
    if include:
        names = lazy.select(pl.col("document").unique()).collect()["document"].to_list()
        keep = [n for n in names if any(fnmatch.fnmatch(n.lower(), g.lower()) for g in include)]
        lazy = lazy.filter(pl.col("document").is_in(keep))
    if formats:
        lazy = lazy.filter(pl.col("source_format").is_in(formats))
    if where:
        lazy = pl.SQLContext(store=lazy).execute(f"SELECT * FROM store WHERE {where}")
    return lazy
