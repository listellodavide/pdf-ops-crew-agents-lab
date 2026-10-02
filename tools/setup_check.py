#!/usr/bin/env python3
"""Setup check, run before the workshop and at 09:45 on Day 1.

    python tools/setup_check.py            # local + cloud
    python tools/setup_check.py --offline  # local only (Python, packages, Tesseract, tests)

Every line is OK, WARN (works, but something is missing) or FAIL (fix before the workshop).
Paste the output in the workshop channel if anything fails.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRED = ["pymupdf", "polars", "pyarrow", "numpy", "pydantic", "langgraph", "azure.ai.projects",
            "azure.search.documents", "azure.identity", "dotenv", "pytest"]
OPTIONAL = ["sentence_transformers", "semantic_kernel", "autogen_agentchat"]
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:<4}] {name}{': ' + detail if detail else ''}", flush=True)


def local() -> None:
    v = sys.version_info
    check("Python", "OK" if v >= (3, 14) else "FAIL", f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
            check(f"package {mod}", "OK")
        except ImportError as e:
            check(f"package {mod}", "FAIL", f"{e}; run pip install -r requirements.txt")
    for mod in OPTIONAL:
        try:
            importlib.import_module(mod)
            check(f"optional {mod}", "OK")
        except ImportError:
            check(f"optional {mod}", "WARN", "not installed (only needed for some stretch goals)")
    tesseract = shutil.which("tesseract") or (Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR" / "tesseract.exe")
    check("Tesseract OCR", "OK" if tesseract and Path(str(tesseract)).exists() else "WARN",
          "needed only for scanned PDFs (winget install UB-Mannheim.TesseractOCR / brew install tesseract)")
    env = os.environ.copy()
    env["PDFMEM_SOLUTIONS"] = "1"
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    last = (run.stdout.strip().splitlines() or ["no output"])[-1]
    check("reference tests (simulator)", "OK" if run.returncode == 0 else "FAIL", last)


def cloud_checks() -> None:
    from pdfmem import cloud
    for var in ("PROJECT_ENDPOINT", "MODEL_DEPLOYMENT_NAME", "PDFMEM_TEAM"):
        check(f".env {var}", "OK" if os.environ.get(var) else "FAIL", os.environ.get(var, "missing"))
    try:
        cloud.credential().get_token("https://ai.azure.com/.default")
        check("Entra ID sign-in", "OK", "token for ai.azure.com")
    except Exception as e:
        check("Entra ID sign-in", "FAIL", f"{type(e).__name__}: run az login")
        return
    try:
        client = cloud.project().get_openai_client()
        reply = client.responses.create(model=os.environ["MODEL_DEPLOYMENT_NAME"], input="Reply with OK.")
        check("Foundry model", "OK", reply.output_text.strip()[:20])
    except Exception as e:
        check("Foundry model", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")
    try:
        emb = client.embeddings.create(model=os.environ.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
                                       input=["setup check"])
        check("Foundry embeddings", "OK", f"{len(emb.data[0].embedding)} dimensions")
    except Exception as e:
        check("Foundry embeddings", "WARN", f"{type(e).__name__}: use --model e5base for the snapshot")
    try:
        names = list(cloud.index_client().list_index_names())
        check("Azure AI Search", "OK", f"{cloud.search_endpoint()} ({len(names)} indexes)")
    except Exception as e:
        check("Azure AI Search", "FAIL", f"{type(e).__name__}: {str(e)[:160]} (role: Search Index Data Contributor "
                                         "+ Search Service Contributor)")
    try:
        stores = list(cloud.project().beta.memory_stores.list(limit=1))
        check("Foundry memory store (preview)", "OK", f"{len(stores)} visible")
    except Exception as e:
        check("Foundry memory store (preview)", "WARN", f"{type(e).__name__}: Lab 2 Block 1 Part B uses the simulator")


def main() -> int:
    p = argparse.ArgumentParser(description="Workshop setup check.")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    local()
    if not args.offline:
        cloud_checks()
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (" - fix the FAIL lines" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
