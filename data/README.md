# Workshop corpus

Real, published PDFs only (no generated documents). `fetch_corpus.py` downloads them from the
official URLs in `manifest.csv` into `data/inbox/`, which git ignores: the documents stay the
publishers' property and are never redistributed with this repository.

| Category | Why it is in the corpus |
|---|---|
| NHTSA Part 573 recall reports | Short, structured filings: campaign numbers, affected models, dates, remedies. Good for facts and the graph lab |
| NHTSA technical service bulletins | Dealer-only content: the permission-aware examples |
| Daimler Truck ESG Factbook FY2024 and FY2025 | The same report two years in a row: recency, fact updates, lessons |
| Scania sustainability tables and report | Long, table-heavy documents: chunking, compression, noise |
| Tesla Model 3 owner's manual | 300 pages that mention "tire", "brake" and "engine" everywhere: keyword noise |

Trainer checklist (see the backlog, epic E1):
1. `python data/fetch_corpus.py --pin` two weeks before; commit only the updated `manifest.csv`.
2. Run Phase 1 and 2a (see the root README) and check every PDF produced text.
3. Write 20 to 30 golden questions in `data/golden.jsonl` from the real documents, following
   `golden.example.jsonl` (mix the types fact, recent, entity, multi-hop). Each answer must be
   checked by a person against the page it cites.
4. Re-run `fetch_corpus.py` the day before: a `[CHANGED]` line means a golden answer may be stale.
