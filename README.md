# PDF Ops Crew Agents Lab: memory architectures for PDF agents on Microsoft Foundry

Hands-on labs for the Adobe Romania Agentic AI workshop (2 days, run twice: Group 1 on days 1-2,
Group 2 on days 3-4, up to 16 people each). The labs continue
[foundry-multiagent-doc-lab](https://github.com/listellodavide/foundry-multiagent-doc-lab): same
kind of documents, now with the question every production agent hits next: **what should it remember,
where, and for how long?**

## Architecture

```
LOCAL (laptop)                                   CLOUD (Microsoft Foundry project + Azure AI Search)
--------------                                   ---------------------------------------------------
data/inbox/*.pdf      real PDFs (data/manifest.csv)
   | Phase 1  pipeline/pdf-compress-extractor.py
pdf-inbox-ingestion/  text, layout JSON, metadata
   | Phase 2a pipeline/embedding-builder.py  ---- Foundry embeddings (text-embedding-3-small)
embeddings/<model>/*.parquet   INTERMEDIATE snapshot: immutable, cheap to rebuild from
   | Phase 3  pdfmem (this repo)  --- optimize ---> pdfmem-<team>-chunks-*    semantic memory (RAG base)
                                                   pdfmem-<team>-memories   facts, edges, episodes,
                                                                            lessons, notes, audit
                                                   Foundry memory store     chat summary, user profile,
                                                                            procedural (preview)
                                                   Foundry Agent Service    prompt agent using the memory
```

Parquet is the intermediate memory only: it is append-only and has no update, delete, expiry or
audit. The resident memory, the one agents read and write, lives in the cloud.

| Memory type | Lab | Resident in | Literature |
|---|---|---|---|
| Semantic (chunks), compressed vectors | 1.1, 1.2 | Azure AI Search | RAG |
| Multi-dimensional recall (meaning, keywords, time, entities) | 1.2 | AI Search hybrid + scoring profile | IMDMR |
| Structured facts with versions | 1.3 | AI Search (memories) | Mem0 |
| Observation log / chat summary | 2.1 | agent context / Foundry memory store | Mastra observational memory |
| Episodic + procedural (lessons) | 2.2 | AI Search (memories) + Foundry memory store | Hindsight (retain, recall, reflect) |
| Knowledge graph + hybrid router + governance | 2.3 | AI Search (memories) | Zep, hybrid architectures |

## Setup: Windows 11 or macOS Tahoe, Python 3.14

```powershell
# Windows 11 (PowerShell)
winget install Python.Python.3.14 Microsoft.AzureCLI UB-Mannheim.TesseractOCR
git clone https://github.com/listellodavide/pdf-ops-crew-agents-lab.git
cd pdf-ops-crew-agents-lab
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env        # then edit: project endpoint, deployments, PDFMEM_TEAM
az login
python tools\setup_check.py
```

```bash
# macOS Tahoe (zsh)
brew install python@3.14 azure-cli tesseract
git clone https://github.com/listellodavide/pdf-ops-crew-agents-lab.git
cd pdf-ops-crew-agents-lab
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env          # then edit
az login
python tools/setup_check.py
```

Every package in `requirements.txt` ships Python 3.14 wheels for Windows x64 and macOS arm64.
Roles needed (assigned by the trainer to the workshop Entra group): **Foundry User** on the Foundry
project, **Search Index Data Contributor** and **Search Service Contributor** on the AI Search service.

## Run the pipeline (trainer or Lab 1 Block 1)

```powershell
python data\fetch_corpus.py                                         # official PDFs -> data\inbox
python pipeline\pdf-compress-extractor.py data\inbox --output pdf-inbox-ingestion
python pipeline\embedding-builder.py pdf-inbox-ingestion --model foundry-te3s
python -m pdfmem.optimize --variants none scalar binary            # Lab 1 Block 1
```

## Labs

See [labs/README.md](labs/README.md): rules, schedule, and one page per 90-minute block.
You edit `pdfmem/` (13 TODOs in total). `solutions/pdfmem/` is the reference; if you are stuck,
`python tools/checkpoint.py <block>` restores a block and keeps yours as `*.mine`.

```powershell
pytest tests/test_lab1_block1.py          # your code
$env:PDFMEM_SOLUTIONS=1; pytest           # the reference (trainers)
```

## Offline contingency

If Azure is unreachable for a team, `PDFMEM_BACKEND=simulator` runs every block against an
in-process stand-in of AI Search and the Foundry memory store (`pdfmem/simulator.py`), and
`--model hash512` builds the snapshot without any model download. It is meant to keep a team
working, not to replace the cloud architecture being taught: sizes and latency are estimates.

## Repository

```
pipeline/     Phase 1 and 2a scripts given to participants (see pipeline/README.md)
pdfmem/       Phase 3, participant version with TODOs (generated by tools/make_starters.py)
solutions/    reference implementation (trainers edit here, then regenerate pdfmem/)
labs/         block handouts, notes template
tests/        per-block tests + synthetic fixtures (fictional, test only)
data/         corpus manifest and fetcher, golden question template
docs/         backlog (epics, user stories, tasks)
tools/        setup check, checkpoint, starter generator
```
