"""Paths and settings, overridable with environment variables (or a .env file)."""

from __future__ import annotations

import os
from pathlib import Path

try:                                    # optional: python-dotenv reads .env in the repo root
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pipeline").is_dir())   # works from solutions/ too
MODELS_CONFIG = REPO / "pipeline" / "configs" / "models.json"

INGESTION_DIR = Path(os.environ.get("PDFMEM_INGESTION", "pdf-inbox-ingestion"))   # Phase 1 output
STORE_DIR = Path(os.environ.get("PDFMEM_STORE", str(INGESTION_DIR / "embeddings")))  # Phase 2a store
MEMORY_DIR = Path(os.environ.get("PDFMEM_MEMORY", "memory"))                     # what the labs write
MODEL = os.environ.get("PDFMEM_MODEL", "e5base")                                 # key in models.json
GOLDEN = Path(os.environ.get("PDFMEM_GOLDEN", "data/golden.jsonl"))

FIXTURES = REPO / "tests" / "fixtures"
