#!/usr/bin/env python3
"""Download the workshop corpus from official sources into data/inbox/ (never committed).

    python data/fetch_corpus.py                 # everything in data/manifest.csv
    python data/fetch_corpus.py --category recall sustainability
    python data/fetch_corpus.py --pin           # write the SHA-256 of each file back to the manifest

The PDFs stay copyrighted by their publishers; they are downloaded for training use and are not
redistributed with the repository. A file whose SHA-256 no longer matches the pinned value is
reported, because publishers replace documents and the golden answers would drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST, INBOX = HERE / "manifest.csv", HERE / "inbox"
HEADERS = {"User-Agent": "Mozilla/5.0 (workshop corpus fetcher; contact: trainer)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", nargs="*", help="recall, service-bulletin, sustainability, owner-manual")
    p.add_argument("--pin", action="store_true", help="store the SHA-256 of downloaded files in the manifest")
    args = p.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    INBOX.mkdir(exist_ok=True)
    failed = 0
    for row in rows:
        if args.category and row["category"] not in args.category:
            continue
        target = INBOX / row["file"]
        if not target.exists():
            try:
                request = urllib.request.Request(row["url"], headers=HEADERS)
                with urllib.request.urlopen(request, timeout=120) as response:
                    target.write_bytes(response.read())
            except Exception as e:                       # network, 403, moved document
                failed += 1
                print(f"[FAIL] {row['file']}: {e}")
                continue
        digest = sha256(target)
        if row["sha256"] and row["sha256"] != digest:
            print(f"[CHANGED] {row['file']}: publisher replaced the document (update golden answers)")
        if args.pin:
            row["sha256"] = digest
        print(f"[OK] {row['file']} ({target.stat().st_size / 1024:,.0f} KB)")
    if args.pin:
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
