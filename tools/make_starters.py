#!/usr/bin/env python3
"""Generate the participant version of pdfmem/ from the reference in solutions/pdfmem/.

Inside the reference, every exercise is marked like this:

    # TODO 1.1-a: what to do ...
    # >>> solution
    ...reference code...
    # <<< solution

In pdfmem/ the region between the markers becomes `raise NotImplementedError("TODO 1.1-a ...")`.
Files without markers are copied unchanged. Run after editing a solution:

    python tools/make_starters.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE, TARGET = ROOT / "solutions" / "pdfmem", ROOT / "pdfmem"
REGION = re.compile(r"(?P<indent>[ \t]*)# >>> solution\n.*?[ \t]*# <<< solution\n", re.DOTALL)
TODO = re.compile(r"# TODO (\d\.\d-[a-z])")


def strip(text: str) -> str:
    out, last = [], 0
    for m in REGION.finditer(text):
        todo = TODO.findall(text[:m.start()])
        label = todo[-1] if todo else "exercise"
        out.append(text[last:m.start()])
        out.append(f'{m.group("indent")}raise NotImplementedError("TODO {label}: see the comment above")\n')
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def main() -> int:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()
    exercises = 0
    for src in sorted(SOURCE.glob("*.py")):
        text = src.read_text(encoding="utf-8")
        exercises += len(REGION.findall(text))
        (TARGET / src.name).write_text(strip(text), encoding="utf-8")
    print(f"pdfmem/ generated from solutions/pdfmem/: {exercises} exercises stubbed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
