"""Golden questions, hit@k / MRR / latency / context tokens, and the team scoreboard.

Golden file (JSON Lines), written by the trainer for the real corpus (see data/golden.example.jsonl):
    {"id": "q1", "question": "...", "expected_documents": ["x.pdf"], "expected_answer_contains": ["125 Nm"],
     "type": "fact|recent|entity|multi-hop"}

Every architecture the teams build is measured the same way and appended to
scoreboard/<team>.csv, which becomes the pros/cons table at the end of each lab.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from pdfmem import cloud, config
from pdfmem.memory import MemoryItem, now_iso
from pdfmem.text import estimate_tokens


def load_golden(path: Path = config.GOLDEN) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def hit(items: list[MemoryItem], expected: list[str]) -> int | None:
    """1-based rank of the first item from an expected document, or None."""
    wanted = {e.lower() for e in expected}
    for rank, item in enumerate(items, start=1):
        if (item.document or "").lower() in wanted:
            return rank
    return None


def evaluate(recall: Callable[[str], list[MemoryItem]], golden: list[dict], k: int = 3) -> dict:
    ranks, latencies, tokens, per_question = [], [], [], []
    for q in golden:
        started = time.perf_counter()
        items = recall(q["question"])
        latencies.append((time.perf_counter() - started) * 1000)
        rank = hit(items[:k], q["expected_documents"])
        ranks.append(rank)
        tokens.append(sum(estimate_tokens(i.text) for i in items[:k]))
        per_question.append({"id": q["id"], "type": q.get("type"), "rank": rank,
                             "top": items[0].citation if items else None})
    n = max(len(golden), 1)
    return {
        "hit_at_k": round(sum(1 for r in ranks if r) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "context_tokens": round(statistics.mean(tokens), 1) if tokens else None,
        "k": k, "questions": per_question,
    }


def record(architecture: str, result: dict, notes: str = "") -> Path:
    path = Path("scoreboard") / f"{cloud.team()}.csv"
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "team", "architecture", "hit_at_k", "mrr", "p50_ms", "context_tokens", "notes"])
        w.writerow([now_iso(), cloud.team(), architecture, result["hit_at_k"], result["mrr"], result["p50_ms"],
                    result["context_tokens"], notes])
    return path


def main() -> int:
    from pdfmem.semantic import MODES, SemanticMemory
    p = argparse.ArgumentParser(description="Compare recall modes on the golden questions.")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--variant", default=None, help="chunks index variant: none, scalar or binary")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--golden", default=str(config.GOLDEN))
    args = p.parse_args()
    golden = load_golden(Path(args.golden))
    memory = SemanticMemory(args.variant)
    print(f"{'mode':<10}{'hit@' + str(args.k):>8}{'MRR':>7}{'p50 ms':>9}{'ctx tok':>9}")
    for mode in args.modes:
        result = evaluate(lambda q, m=mode: memory.recall(q, k=10, mode=m), golden, args.k)
        record(f"chunks-{memory.variant}/{mode}", result)
        print(f"{mode:<10}{result['hit_at_k']:>8.2f}{result['mrr']:>7.2f}{result['p50_ms']:>9.1f}"
              f"{result['context_tokens']:>9.0f}")
    print(f"\nscoreboard: scoreboard/{cloud.team()}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
