# Lab 1, Block 2: recall modes on the resident memory (90 min)

**Memory type:** semantic memory, read path. Same index, five ways to ask it.

| Mode | Service feature | Good at | Weak at |
|---|---|---|---|
| keyword | BM25 | part numbers, campaign ids, exact terms | synonyms, paraphrases; "tire" is everywhere |
| vector | kNN on compressed vectors | meaning, paraphrases | exact codes, numbers |
| hybrid | both, Reciprocal Rank Fusion | robust default | costs two searches |
| semantic | hybrid + semantic ranker | ordering the top results | extra latency, tier and quota limits |
| fresh | hybrid + scoring profile (freshness, entity tags) | "latest" questions, entity-heavy questions | needs good dates and entities at load time |

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 1.2-a | `pdfmem/semantic.py` `search_kwargs` | The `SearchClient.search()` arguments for each mode |
| 1.2-b | `pdfmem/semantic.py` `documents_filter` | An OData `search.in` filter on document names |

Done when `pytest tests/test_lab1_block2.py` is green.

## Break (10 min)

## Part B (40 min): compare on the golden questions

```powershell
python -m pdfmem.evaluate --modes keyword vector hybrid semantic fresh
python -m pdfmem.semantic "tire pressure" --mode keyword
python -m pdfmem.semantic "tire pressure" --mode hybrid --document tesla-model3-owners-manual-eu.pdf
```

1. Which mode wins hit@3? Which wins MRR? Which questions does each one miss (see `questions` in
   the evaluation output)? Write the row for every mode in the scoreboard (done automatically).
2. Noise: run "tire" in keyword mode. Then add `--document`: how much of the noise was a *filter*
   problem, not a *ranking* problem?
3. Stretch (pick one):
   - Change the freshness boost or duration in `schema.scoring_profiles()`, re-run Block 1.1 for
     `scalar` only, compare `fresh` again.
   - Add a filter for `created_at ge 2025-01-01T00:00:00Z` to the "latest" questions only.
   - Permission-aware recall: give dealer-only documents (service bulletins) an `access` value and
     filter on it. What would you change in the index schema? (Discuss: security trimming with Entra groups.)
