# PDF inbox ingestion: Phase 1 and Phase 2

Scripts that turn a folder of PDFs (sustainability reports, car and truck spare-parts catalogs,
service bulletins, brochures) and tabular exports (CSV, ORC, Parquet, Excel, ...) into a local
embedding store for the AI labs.

```
project root
  *.pdf ──► Phase 1  pdf-compress-extractor.py ──► pdf-inbox-ingestion\
                                                     <stem>.txt             text for the LLM, [[page N]] markers
                                                     <stem>.layout.json     blocks, lines, spans, fonts, bboxes, tables
                                                     <stem>.compressed.pdf  smaller PDF, scans made searchable
                                                     <stem>.metadata.json   hashes, sizes, OCR decisions, warnings
  *.csv *.orc *.parquet *.xlsx ... ─┐
            Phase 2a embedding-builder.py ──► pdf-inbox-ingestion\embeddings\<model>\*.parquet
            Phase 2b pdf-semantic-ranker.py "query" ──► ranked documents or rows, with pages and attributes
```

## Setup on Windows 11

```powershell
cd C:\path\to\project
py -3.14 -m venv .venv            # 3.12 and 3.13 work too
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

winget install UB-Mannheim.TesseractOCR   # only needed for scanned pages
```

The Tesseract installer lets you tick extra languages (for example Romanian `ron`, German `deu`,
Italian `ita`). The script finds `C:\Program Files\Tesseract-OCR\tessdata` automatically; otherwise
set `TESSDATA_PREFIX` or pass `--tessdata`.

The first Phase 2a run downloads `intfloat/multilingual-e5-base` (about 1.1 GB) from Hugging Face.
On a network that blocks Hugging Face, download it once elsewhere and pass the folder with `--model`.
`intfloat/multilingual-e5-small` is a faster alternative (about 470 MB).

## Phase 1: extract and compress

```powershell
python .\pdf-compress-extractor.py                       # every PDF in the project root
python .\pdf-compress-extractor.py .\catalog.pdf --mode text
python .\pdf-compress-extractor.py --ocr-language "eng+deu+ron"
python .\pdf-compress-extractor.py --force               # reprocess unchanged files
```

| Option | Default | Effect |
|---|---|---|
| `--mode` | `all` | `text` writes .txt, layout and metadata; `compress` writes only the PDF |
| `--ocr` | `auto` | `auto` OCRs only pages with under `--min-chars` native characters and at least `--min-image` image coverage; `never`; `always` |
| `--ocr-dpi` | 300 | render resolution for Tesseract |
| `--image-dpi`, `--jpeg-quality` | 150, 70 | image downsampling in the compressed PDF |
| `--no-tables` | off | keep ruled tables as plain lines instead of Markdown |
| `--keep-boilerplate` | off | keep repeated headers and footers on every page |
| `--pretty` | off | indented layout JSON (about 3 times larger) |

What it does, and why:

* Native text is never replaced by OCR. Part numbers, torque values and emission figures in a
  born-digital PDF are exact; OCR would add errors.
* Scanned pages get OCR and an invisible text layer in the compressed PDF, so they become searchable
  in Acrobat and in Phase 2. A page that OCRs as noise is retried turned 90 and 270 degrees; a
  sideways scan is turned upright in the compressed PDF. Pages with low OCR quality are listed in
  `warnings`.
* Lines repeated in the top or bottom 8% of at least half of the pages (running headers, page
  numbers) are kept once and then removed. They are listed in `removed_header_footer_lines`.
* Ruled tables (parts lists) become Markdown tables, one row per part, so the LLM sees which
  description and quantity belong to which part number.
* An unchanged PDF (same SHA-256, same settings) is skipped on the next run.

## Phase 2a: build the embedding store (Parquet + Polars)

```powershell
python .\embedding-builder.py                                  # Phase 1 output folder
python .\embedding-builder.py .\data\parts.csv --text-column description --text-column vehicle
python .\embedding-builder.py .\pdf-inbox-ingestion .\data --model e5small
```

| Input | Formats | How it is read |
|---|---|---|
| Text | `.txt .md .markdown .rst .log` | Phase 1 `.txt` keeps page numbers; others are chunked by paragraph. UTF-8, falls back to cp1252 |
| Tables | `.csv .tsv .parquet .orc .jsonl .ndjson .json .arrow .feather .ipc .xlsx .xls` | Batches of `--batch-rows` (default 10,000) through Polars; ORC stripe by stripe through PyArrow |

* Tables: one chunk per row (several if the text is longer than `--chunk-size`). Choose the text with
  `--text-column`; repeat it to combine columns as `description: ...` / `vehicle: ...` lines. Without
  it, or if a file lacks that column, the longest text column is used. The other columns are kept
  as JSON in `attributes`, so a hit still shows part number, price and stock.
* Store: `pdf-inbox-ingestion\embeddings\<model>\<source file>.parquet`, zstd-compressed, one file per
  source. Columns: `id, document, source, source_format, record, chunk, page_start, page_end, text,
  n_chars, attributes, embedding`. `embedding` is `Array(Float32, dim)` and L2-normalized.
* The model, prefixes, chunk settings and source SHA-256 are stored in the Parquet file metadata.
  An unchanged source is skipped; `--force` rebuilds. Files are written to `.partial` first and
  renamed at the end, so an interrupted run leaves no half file.
* Models are keys in `configs\models.json` (`e5small`, `e5base` default, `e5large`, `paraphrase`,
  `gte`, `qwen`) or any sentence-transformers name or local folder. GPU (`cuda`, `mps`) is used when
  present.

## Phase 2b: rank documents or rows

```powershell
python .\pdf-semantic-ranker.py "Sustainability"
python .\pdf-semantic-ranker.py "reduction of carbon emissions" --top 5 --json ranking.json
python .\pdf-semantic-ranker.py "brake pads for Duster" --level chunk --format csv --format orc
python .\pdf-semantic-ranker.py "emissions" --where "source_format = 'pdf-text' AND page_start <= 10"
python .\pdf-semantic-ranker.py "tire" --method keyword
```

* Similarity is the cosine of the angle between query vector q and stored vector d:
  `cos(q, d) = (q . d) / (||q|| x ||d||)`. Vectors are stored unit length, so this equals the dot
  product, but the formula is computed in full. The top-k uses `argpartition`, not a full sort.
* Filters run in Polars before any vector is compared: `--include` (document name pattern),
  `--format`, and `--where` with any SQL condition on the store columns.
* `--level document` (default) groups chunks by document: 0.6 x best chunk + 0.3 x mean of the 3
  best + 0.1 x coverage, after rescaling scores to 0..1. Coverage = the document's chunks in the
  corpus top 50, divided by 5 (max 1). Use it for PDFs and reports.
* `--level chunk` returns the best chunks or table rows with their attributes. Use it for parts
  lists, where every row is its own answer.
* `--method keyword` (BM25) and `--method hybrid` (Reciprocal Rank Fusion, k = 60) use the same
  stored text, so the three methods can be compared on one query.

Measured on the test machine (CPU): 200,000 vectors of 768 dimensions load from Parquet in 2.3 s;
`to_numpy()` is zero-copy; cosine plus top-10 takes 0.35 s.

### Why this corpus is a good lab

"tire", "engine" and "brake" appear in nearly every catalog and brochure, so `--method keyword`
returns long, flat rankings. Compare the three methods on the same query, then add `--include`,
`--format` or `--where` to see how much of the noise a metadata filter removes before any ranking.

## Tested

Linux, Python 3.11, PyMuPDF 1.28.2, Tesseract 5.3.4, Polars 1.44, PyArrow 25, sentence-transformers
6.1. Phase 1: born-digital reports with tables, a parts catalog with ruled tables, scans (upright,
scanner-rotated, sideways), an image-heavy brochure. Phase 2a/2b: Phase 1 text plus CSV (25,000
rows, streamed in batches), TSV, Parquet, ORC, JSON Lines, JSON, Arrow IPC, Excel, Markdown and a
cp1252 text file. The semantic path ran with a small local model because Hugging Face was not
reachable from the test machine; ranking quality with E5 still has to be checked on the real data.

## Credits

The store design adapts [brian-ogrady/local-embed-gen](https://github.com/brian-ogrady/local-embed-gen)
(MIT License, Copyright (c) 2025 brian-ogrady): batched input, a JSON model registry and zstd Parquet
output. Changes: more input formats, fixed-size `Array(Float32)` vectors, automatic device instead of
a fixed `mps`, page-aware chunks, skip-if-unchanged, and search. Storage and search follow
[Max Woolf, "The Best Way to Use Text Embeddings Portably is With Parquet and Polars"](https://minimaxir.com/2025/02/embeddings-parquet/).
