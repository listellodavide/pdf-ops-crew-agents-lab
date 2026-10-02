#!/usr/bin/env python3
"""Hackathon checkpoint: copy the reference solution of one or more blocks into pdfmem/.

    python tools/checkpoint.py 1.2          # restore Lab 1 Block 2 (semantic.py)
    python tools/checkpoint.py upto 2.1     # restore every block before Lab 2 Block 1
    python tools/checkpoint.py list

Your own version is kept as pdfmem/<module>.py.mine, so nothing is lost.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {
    "1.1": ["optimize.py"], "1.2": ["semantic.py"], "1.3": ["facts.py"],
    "2.1": ["observe.py"], "2.2": ["episodic.py"], "2.3": ["graph.py", "hybrid.py"],
}


def restore(block: str) -> None:
    for name in BLOCKS[block]:
        target, source = ROOT / "pdfmem" / name, ROOT / "solutions" / "pdfmem" / name
        if target.exists():
            shutil.copy2(target, target.with_name(name + ".mine"))
        shutil.copy2(source, target)
        print(f"[checkpoint] {block}: pdfmem/{name} restored (yours: pdfmem/{name}.mine)")


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "list":
        for block, files in BLOCKS.items():
            print(f"{block}: {', '.join(files)}")
        return 0
    if argv[0] == "upto":
        if len(argv) < 2 or argv[1] not in BLOCKS:
            print(f"usage: checkpoint.py upto <block>, block in {', '.join(BLOCKS)}")
            return 1
        for block in BLOCKS:
            if block == argv[1]:
                break
            restore(block)
        return 0
    for block in argv:
        if block not in BLOCKS:
            print(f"unknown block {block}; use one of {', '.join(BLOCKS)}")
            return 1
        restore(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
