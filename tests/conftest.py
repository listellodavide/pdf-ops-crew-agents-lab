"""Test setup: simulator backend, a hashing-model snapshot of the fixtures, per-block isolation.

    pytest tests/test_lab1_block1.py      # your code in pdfmem/
    PDFMEM_SOLUTIONS=1 pytest             # the reference in solutions/pdfmem/ (trainers)

Each block's tests use only that block's TODOs; shared setup is done here with reference-free
helpers, so a team stuck on Block 1.2 can still validate Block 1.3.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
_WORK = Path(tempfile.mkdtemp(prefix="pdfmem-tests-"))

os.environ.update({
    "PDFMEM_BACKEND": "simulator", "PDFMEM_OFFLINE": "1", "PDFMEM_TEAM": "test",
    "PDFMEM_INGESTION": str(FIXTURES / "inbox-ingestion"), "PDFMEM_STORE": str(_WORK / "store"),
    "PDFMEM_MEMORY": str(_WORK / "memory"), "PDFMEM_MODEL": "hash512",
    "PDFMEM_GOLDEN": str(FIXTURES / "golden.jsonl"), "PDFMEM_CHUNKS_VARIANT": "scalar",
})
os.environ.pop("PROJECT_ENDPOINT", None)
sys.path.insert(0, str(ROOT / "solutions") if os.environ.get("PDFMEM_SOLUTIONS") == "1" else str(ROOT))


@pytest.fixture(scope="session")
def snapshot():
    """Phase 2a output for the fixtures: Parquet store built with the offline hashing model."""
    from pdfmem import config
    from pdfmem import embedstore as es
    spec = es.load_model_spec("hash512", config.MODELS_CONFIG)
    embedder = es.Embedder(spec)
    for txt in sorted((FIXTURES / "inbox-ingestion").glob("*.txt")):
        meta = {"schema": es.SCHEMA_VERSION, "model_key": spec.key, "model_name": spec.model_name,
                "query_prefix": "", "passage_prefix": "", "normalized": True, "source": txt.name,
                "source_format": "pdf-text", "source_sha256": es.sha256_of(txt), "chunk_size": 400,
                "chunk_overlap": 0, "text_columns": []}
        es.write_source(txt, es.store_file(config.STORE_DIR, spec, txt), es.text_rows(txt, 400, 0), embedder, meta)
    lazy, meta = es.open_store(config.STORE_DIR, spec.key)
    return lazy.collect(), meta


def documents_for(frame, dates) -> list[dict]:
    """Reference-free mapping (same contract as TODO 1.1-c) used to load indexes for later blocks."""
    from pdfmem.memory import safe_key
    from pdfmem.text import entities
    return [{"id": safe_key(r["id"]), "text": r["text"], "document": r["document"], "source_format": r["source_format"],
             "page_start": r["page_start"], "page_end": r["page_end"], "created_at": dates.get(r["document"]),
             "entities": sorted(entities(r["text"])), "attributes": r["attributes"],
             "embedding": [float(x) for x in r["embedding"]]} for r in frame.iter_rows(named=True)]


@pytest.fixture()
def fresh_backend():
    from pdfmem import cloud, simulator
    simulator.reset()
    cloud.reset_clients()
    yield
    simulator.reset()


@pytest.fixture()
def loaded_chunks(snapshot, fresh_backend):
    """The team's scalar chunks index loaded with the fixtures, plus an empty memories index."""
    from pdfmem import cloud, schema
    from pdfmem.memory import document_dates
    frame, meta = snapshot
    dims = int(meta["dimension"])
    chunks = cloud.index_name("chunks", "scalar")
    cloud.index_client().create_or_update_index(schema.chunk_index(chunks, dims, "scalar"))
    cloud.search_client(chunks).upload_documents(documents_for(frame, document_dates()))
    cloud.index_client().create_or_update_index(schema.memories_index(cloud.index_name("memories"), dims))
    return meta


@pytest.fixture(scope="session")
def golden():
    return [json.loads(line) for line in (FIXTURES / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line]
