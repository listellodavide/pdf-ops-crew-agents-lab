# Lab 1, Block 1: optimize the Parquet snapshot and load it as resident memory (90 min)

**Memory type:** semantic memory (RAG as the base layer).
**Where it lives:** Azure AI Search, connected to your Foundry project. Parquet is only the
intermediate snapshot produced by Phase 2a.

```
data/inbox/*.pdf --Phase 1--> pdf-inbox-ingestion/*.txt --Phase 2a--> embeddings/<model>/*.parquet
                                                                          |  (this block)
                                              optimize: short, exact, near-duplicates
                                                                          v
                                   Azure AI Search  pdfmem-<team>-chunks-{none|scalar|binary}
```

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 1.1-a | `pdfmem/optimize.py` `drop_near_duplicates` | Greedy de-duplication with one matrix product per row |
| 1.1-b | `pdfmem/optimize.py` `create_chunk_index` | Create the index for a compression variant (read `pdfmem/schema.py` first) |
| 1.1-c | `pdfmem/optimize.py` `to_documents` | Map a snapshot row to the index document (date, entities, vector) |

Done when `pytest tests/test_lab1_block1.py` is green.

Questions to discuss while you code:
- Why compare against *kept* chunks only, and what does the order of the rows change?
- `stored=False` on the vector field: what do you lose? (Hint: you cannot read vectors back.)

## Break (10 min)

## Part B (40 min): measure on the real corpus

```powershell
python pipeline\pdf-compress-extractor.py data\inbox --output pdf-inbox-ingestion      # trainer may have done this
python pipeline\embedding-builder.py pdf-inbox-ingestion --model foundry-te3s
python -m pdfmem.optimize --variants none scalar binary --model foundry-te3s
python -m pdfmem.evaluate --variant none --modes vector
python -m pdfmem.evaluate --variant scalar --modes vector
python -m pdfmem.evaluate --variant binary --modes vector
```

1. Fill the table: reduction from the optimizer, `storage_size` and `vector_index_size` per variant
   (`memory/lab1_block1.json`), hit@3 and MRR per variant.
2. Stretch (pick one):
   - lower `--threshold` to 0.93: how many chunks go, and does hit@3 move?
   - raise `--min-chars` to 150: which questions break, and why?
   - run Phase 2a again with `--model e5base`: same corpus, different vector space; compare hit@3.
3. Delete the variants you will not use (Basic tier: 15 indexes for the whole group):
   `python -m pdfmem.cleanup --keep scalar`.

## What to write in your notes
- Scalar quantization makes the vector index about 4x smaller; binary about 32x. With rescoring the
  originals are kept on disk, so *storage* does not shrink the same way. Which one is the bottleneck
  on your tier: memory or disk?
- Duplicate chunks are not harmless: they take two of your top-3 slots with the same content.
