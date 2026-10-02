# Lab 1, Block 3: fact memory, Mem0-style (90 min)

**Memory type:** semantic long-term memory as structured facts (subject, predicate, object) with
provenance and versions. **Where it lives:** `pdfmem-<team>-memories`, kind `fact`.

```
chunk --extract (Foundry structured output, or rules offline)--> facts
facts --consolidate against the stored version: ADD / UPDATE / NOOP--> memories index (history kept)
question --entities--> facts about those entities (a few dozen tokens instead of 3 chunks)
```

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 1.3-a | `pdfmem/facts.py` `consolidate` | Mem0's update rule with dates: a stale report must not overwrite a newer one |
| 1.3-b | `pdfmem/facts.py` `FactMemory.recall` | Facts about the question's entities, keyword fallback |

Done when `pytest tests/test_lab1_block3.py` is green.

## Break (10 min)

## Part B (40 min): extract from the real corpus and compare

```powershell
python -m pdfmem.facts extract --documents daimler-truck-esg-factbook-2024.pdf daimler-truck-esg-factbook-2025.pdf
python -m pdfmem.facts extract --documents nhtsa-rclrpt-23V083.pdf nhtsa-rclrpt-23V111.pdf
python -m pdfmem.facts ask "latest scope 1 and 2 emissions"
python -m pdfmem.facts ask "Which vehicles are affected by 23V083?"
```

1. How many ADD / UPDATE / NOOP? Open two UPDATEs: is the history right?
2. Answer three golden questions from facts only and from chunks only. Compare context tokens
   (facts: one line each) and correctness.
3. With `PDFMEM_OFFLINE=1` the extractor uses rules; without it, the Foundry model. Compare the
   number and quality of facts on the same two documents: precision vs coverage vs cost.
4. Stretch: entity resolution. "Scope 1+2 emissions" and "CO2 emissions (scope 1 and 2)" are the
   same metric with two subjects. Propose a normalization rule (or a model call) and where it runs.

## Notes
Fact memory is cheap to read and easy to audit, but extraction is lossy and expensive to write.
Where would a wrong fact hurt Adobe more than a wrong chunk?
