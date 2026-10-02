#!/usr/bin/env python3
"""Workshop backlog: 4 epics x 5 user stories x 5 tasks, written to docs/backlog.md and
docs/backlog.csv (Azure Boards hierarchical import: Work Item Type, Title 1/2/3).

Edit the data below, then:  python tools/build_backlog.py
Estimates are hours. Owners: Trainer, Co-trainer, Azure admin (Microsoft), Adobe coordinator.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (title, description, [ (story, acceptance criteria, [ (task, hours, owner), x5 ] ), x5 ])
BACKLOG = [
    ("E1 Corpus, snapshot and golden set",
     "Real, published PDFs flow through Phase 1 and 2a into a Parquet snapshot that every team can load, "
     "with golden questions written and checked by a person.",
     [
         ("As a trainer, I want a corpus of real automotive and sustainability PDFs, so that participants work on "
          "documents with the same noise and repetition as Adobe's inbox.",
          "13+ PDFs from official sources listed in data/manifest.csv; SHA-256 pinned; nothing generated; "
          "copyrighted files never committed.",
          [("Shortlist recall reports, service bulletins, two years of one ESG report, one owner's manual", 3, "Trainer"),
           ("Run data/fetch_corpus.py --pin and commit only the manifest", 1, "Trainer"),
           ("Check licence and terms of each source; note download-only documents", 2, "Trainer"),
           ("Add 2-3 documents from Adobe's own inbox if Adobe provides anonymised ones", 2, "Adobe coordinator"),
           ("Re-run the fetcher the day before each group; handle [CHANGED] documents", 1, "Co-trainer")]),
         ("As a participant, I want the PDFs already extracted to text and layout, so that the labs start from memory "
          "design, not from parsing.",
          "pdf-inbox-ingestion/ contains .txt, .layout.json, .metadata.json for every PDF; OCR pages flagged; "
          "no WARN line about missing text.",
          [("Run pdf-compress-extractor.py on the corpus with OCR auto", 1, "Trainer"),
           ("Review metadata.json warnings and OCR quality for scanned pages", 2, "Trainer"),
           ("Check repeated headers/footers removal on the two biggest reports", 1, "Co-trainer"),
           ("Record page counts, sizes and compression ratios for the slides", 1, "Co-trainer"),
           ("Zip the Phase 1 output as a fallback download for teams", 1, "Trainer")]),
         ("As a participant, I want a Parquet embedding snapshot built with the same model the cloud uses, so that "
          "my queries and the stored vectors are in one vector space.",
          "embeddings/foundry-te3s/*.parquet built with the Foundry embedding deployment; model and source hash in "
          "file metadata; e5base snapshot built as an offline alternative.",
          [("Build the snapshot with --model foundry-te3s", 1, "Trainer"),
           ("Build the e5base snapshot for the offline comparison", 1, "Trainer"),
           ("Measure tokens and cost of the embedding run; put it on the cost slide", 1, "Trainer"),
           ("Verify skip-if-unchanged by re-running the builder", 0.5, "Co-trainer"),
           ("Publish the snapshot to the shared storage the teams download from", 1, "Azure admin")]),
         ("As a trainer, I want 20-30 golden questions with checked answers, so that every architecture is scored "
          "on the same ground truth.",
          "data/golden.jsonl with fact, recent, entity and multi-hop questions; each answer checked by a person "
          "against the cited page; at least 4 'latest' questions spanning the two ESG years.",
          [("Draft 30 questions across the four types from the real documents", 4, "Trainer"),
           ("Verify every expected answer and document against the page", 3, "Co-trainer"),
           ("Add 3 multi-hop questions (recall -> model -> part) for Lab 2 Block 3", 2, "Trainer"),
           ("Run evaluate.py with the reference to get the baseline scoreboard", 1, "Trainer"),
           ("Review the question list with Adobe for domain relevance", 1, "Adobe coordinator")]),
         ("As a trainer, I want the reference implementation and tests to pass on the real corpus, so that no lab "
          "fails on the day for reasons unrelated to the exercise.",
          "PDFMEM_SOLUTIONS=1 pytest green; every block's CLI run end to end on Azure with the real corpus; "
          "timings recorded per block.",
          [("Dry run all six blocks on Azure with the reference (checkpoint upto 2.3)", 4, "Trainer"),
           ("Dry run the same with the simulator backend (contingency)", 1, "Co-trainer"),
           ("Time Part A and Part B of each block with a colleague who did not write the code", 3, "Co-trainer"),
           ("Fix or simplify any TODO that takes more than 30 minutes", 3, "Trainer"),
           ("Tag the repository version used for each group", 0.5, "Trainer")]),
     ]),
    ("E2 Lab 1: from Parquet to resident memory",
     "Participants turn the snapshot into resident semantic memory and fact memory in Azure AI Search, and "
     "measure size, recall quality, latency and tokens for each design.",
     [
         ("As a participant, I want to optimize the snapshot before loading it, so that the resident memory "
          "holds no fragments or duplicates.",
          "TODO 1.1-a green; optimize report shows short/exact/near counts; teams can explain the threshold choice.",
          [("Write the Block 1.1 handout (Part A, Part B, notes)", 2, "Trainer"),
           ("Prepare the slide on duplicates taking top-k slots", 1, "Trainer"),
           ("Pick two threshold values that visibly change the result on the corpus", 1, "Co-trainer"),
           ("Prepare the debrief questions on order and threshold", 0.5, "Trainer"),
           ("Check test_lab1_block1 timings on a Windows and a Mac laptop", 1, "Co-trainer")]),
         ("As a participant, I want to compare uncompressed, scalar and binary quantized indexes, so that I can "
          "choose vector compression with numbers.",
          "Three indexes per team created and measured (storage, vector index size, hit@3); unused variants "
          "deleted with cleanup --keep.",
          [("Confirm AI Search tier and index limits for 4 teams x 4 indexes", 1, "Azure admin"),
           ("Explain rescoring and preserveOriginals on one slide", 1, "Trainer"),
           ("Record the reference numbers for the three variants", 1, "Trainer"),
           ("Prepare the 'memory vs disk' discussion", 0.5, "Trainer"),
           ("Script the cleanup between Part B and Block 1.2", 0.5, "Co-trainer")]),
         ("As a participant, I want to query the memory in keyword, vector, hybrid, semantic and fresh modes, so "
          "that I see which questions each mode answers.",
          "TODO 1.2-a and 1.2-b green; five scoreboard rows per team; every team names one question each mode misses.",
          [("Write the Block 1.2 handout and mode comparison table", 2, "Trainer"),
           ("Check semantic ranker availability and quota on the Search tier", 0.5, "Azure admin"),
           ("Prepare the 'tire' noise demo with and without a document filter", 1, "Trainer"),
           ("Tune the freshness and tag boosts on the real corpus", 2, "Trainer"),
           ("Prepare the permission-aware stretch (access field, Entra groups)", 2, "Co-trainer")]),
         ("As a participant, I want facts with versions instead of chunks for exact questions, so that the agent "
          "reads one line with provenance and the newest value wins.",
          "TODO 1.3-a and 1.3-b green; facts extracted from both ESG years; at least one UPDATE with correct history.",
          [("Write the Block 1.3 handout", 2, "Trainer"),
           ("Run model-based extraction on the ESG factbooks and review 20 facts by hand", 2, "Trainer"),
           ("Prepare the rules-vs-model comparison (count, precision, cost)", 1, "Co-trainer"),
           ("Prepare the entity-resolution stretch example", 1, "Trainer"),
           ("Estimate model tokens for extraction per team; check TPM quota", 1, "Azure admin")]),
         ("As a team, I want a scoreboard and a pros/cons table after Lab 1, so that we compare designs with our "
          "own numbers rather than vendor claims.",
          "scoreboard/<team>.csv has rows for every variant and mode; notes table filled for 1.1-1.3; 45-minute "
          "debrief held.",
          [("Build the day-1 scoreboard view (merge team CSVs)", 1, "Co-trainer"),
           ("Prepare the debrief slide template", 1, "Trainer"),
           ("Collect one insight per team for the Day 2 recap", 0.5, "Co-trainer"),
           ("Compare team numbers with the article's claims on one slide", 1, "Trainer"),
           ("Award the day's points", 0.5, "Trainer")]),
     ]),
    ("E3 Lab 2: agents that remember",
     "Participants give an agent short-term, episodic, procedural and graph memory, route questions across "
     "memories with governance, and expose the memory to Foundry Agent Service and other frameworks.",
     [
         ("As a participant, I want an observation log with a token budget, so that long sessions over big PDFs "
          "stay small and stable.",
          "TODO 2.1-a and 2.1-b green; compression ratio recorded; context never exceeds the budget.",
          [("Write the Block 2.1 handout", 2, "Trainer"),
           ("Prepare the observer/reflector diagram slide", 1, "Trainer"),
           ("Record reference ratios for the rule-based and the model-based observer", 1, "Co-trainer"),
           ("Prepare the budget 400 vs 150 discussion", 0.5, "Trainer"),
           ("Check model latency for per-event observation calls", 0.5, "Co-trainer")]),
         ("As a participant, I want to compare my log with the Foundry memory store, so that I know what managed "
          "memory gives and what it hides.",
          "Memory store created per team; chat summary and user profile memories visible; teams list one thing "
          "each approach kept that the other missed.",
          [("Confirm memory store preview availability in the region and subscription", 1, "Azure admin"),
           ("Create one store per team in advance (chat and embedding models set)", 1, "Azure admin"),
           ("Run the --foundry path once per team in the dry run", 1, "Co-trainer"),
           ("Prepare the managed-vs-own comparison table", 1, "Trainer"),
           ("Plan the fallback if the preview is unavailable (simulator)", 0.5, "Trainer")]),
         ("As a participant, I want the agent to learn lessons from failed episodes, so that it improves without "
          "code or prompt changes.",
          "TODO 2.2-a and 2.2-b green; pass 2 success rate higher than pass 1 on the real golden set; lessons "
          "visible in the index and optionally in Foundry.",
          [("Write the Block 2.2 handout", 2, "Trainer"),
           ("Pick recall settings (mode, k) where pass 1 fails on 'latest' questions", 1, "Trainer"),
           ("Prepare the LangGraph diagram and state explanation", 1, "Trainer"),
           ("Prepare the over-learning counter-example", 1, "Co-trainer"),
           ("Prepare the human-in-the-loop stretch with interrupt", 2, "Trainer")]),
         ("As a participant, I want graph memory and a router across memories, so that multi-hop questions get "
          "answered and simple ones stay cheap.",
          "TODO 2.3-a and 2.3-b green; recall -> model -> part path found on the real corpus; teams compare "
          "hybrid vs semantic-only on five questions.",
          [("Write the Block 2.3 handout", 2, "Trainer"),
           ("Check that the real corpus yields at least one two-hop path", 1, "Trainer"),
           ("Prepare the 'when to use a graph database' slide (Cosmos DB Gremlin, Neo4j)", 1, "Trainer"),
           ("Prepare the governance demo: forget, audit, what remains elsewhere", 1, "Co-trainer"),
           ("Verify the Foundry prompt agent with the function tool in the dry run", 1, "Trainer")]),
         ("As an architect, I want the same memory behind Foundry Agent Service, LangGraph, Semantic Kernel and "
          "AutoGen, so that the framework choice is separate from the memory design.",
          "Adapters run (tests green with optional packages); each team tries one framework and notes what it "
          "gives and costs; final recommendation section filled.",
          [("Verify requirements-frameworks.txt installs on Windows and macOS with Python 3.14", 1, "Co-trainer"),
           ("Prepare the four-framework comparison slide", 1, "Trainer"),
           ("Prepare a Semantic Kernel run against the Foundry deployment (Entra token)", 1, "Trainer"),
           ("Prepare an AutoGen AssistantAgent run with the tool", 1, "Co-trainer"),
           ("Collect each team's recommendation for the closing session", 0.5, "Trainer")]),
     ]),
    ("E4 Delivery: 2 days x 2 groups",
     "The workshop runs twice with up to 16 people per group, remote, on time, with working access, clean "
     "environments between groups, and measured feedback.",
     [
         ("As a participant, I want the theory sessions to explain memory architectures and their trade-offs, "
          "so that the labs make sense.",
          "30-slide deck reviewed; each architecture slide has strengths, weaknesses and when not to use; "
          "article claims cited as claims.",
          [("Build the 30-slide deck", 6, "Trainer"),
           ("Review the deck with a second Microsoft CSA", 1, "Co-trainer"),
           ("Add the team scoreboard and pros/cons template slides", 1, "Trainer"),
           ("Rehearse the 45-minute theory block for each day", 2, "Trainer"),
           ("Share the deck with Adobe one week before", 0.5, "Trainer")]),
         ("As Adobe, I want every participant's laptop and account ready before Day 1, so that no lab time is lost "
          "on setup.",
          "tools/setup_check.py all OK for every participant (offline + cloud); roles assigned to the Entra group; "
          "30-minute setup clinic held.",
          [("Collect participant list and team assignment per group (max 16)", 1, "Adobe coordinator"),
           ("Create the Entra group and assign Foundry User and Search roles", 2, "Azure admin"),
           ("Send setup instructions (Windows 11 and macOS Tahoe, Python 3.14)", 1, "Trainer"),
           ("Run the setup clinic and collect setup_check outputs", 1.5, "Co-trainer"),
           ("Follow up on every FAIL line before Day 1", 2, "Co-trainer")]),
         ("As the trainer, I want quotas and costs sized for 16 people working at once, so that labs do not stall "
          "on HTTP 429 or tier limits.",
          "Model and embedding TPM sized from dry-run token counts x 16; AI Search tier supports 4 teams x up to "
          "4 indexes; cost estimate shared with Adobe.",
          [("Measure tokens per block in the dry run", 1, "Trainer"),
           ("Set deployment TPM and request quota increases if needed", 1, "Azure admin"),
           ("Choose the AI Search tier (Basic vs S1) from the index count", 0.5, "Azure admin"),
           ("Write the cost estimate per group", 1, "Trainer"),
           ("Set a budget alert on the resource group", 0.5, "Azure admin")]),
         ("As the trainer, I want a clean environment for Group 2, so that teams do not see Group 1's memories "
          "and indexes.",
          "cleanup --all run for every Group 1 team; memory stores and agent versions removed; Group 2 team "
          "prefixes (g2-*) set; spot check shows zero leftover indexes.",
          [("Script cleanup for all teams of a group", 1, "Co-trainer"),
           ("Run the cleanup at the end of day 2", 0.5, "Co-trainer"),
           ("Delete leftover agent versions in Foundry", 0.5, "Azure admin"),
           ("Reset scoreboards and notes for Group 2", 0.5, "Co-trainer"),
           ("Apply fixes found during Group 1 and re-tag the repository", 2, "Trainer")]),
         ("As Adobe, I want to measure what the participants learned and leave with a PoC backlog, so that the "
          "workshop leads to a decision.",
          "Feedback form completed by at least 80% of each group; scoreboards and notes archived; a PoC backlog "
          "with success criteria agreed for the most promising use case.",
          [("Prepare the feedback form (content, labs, pace, cloud setup)", 1, "Trainer"),
           ("Run the architecture review and PoC definition session", 1.5, "Trainer"),
           ("Archive team scoreboards and notes", 0.5, "Co-trainer"),
           ("Write the workshop report for Adobe and Microsoft", 3, "Trainer"),
           ("Agree on the next step and the six-month update cadence", 1, "Adobe coordinator")]),
     ]),
]


def validate() -> None:
    assert len(BACKLOG) == 4, "4 epics"
    for epic, _, stories in BACKLOG:
        assert len(stories) == 5, f"{epic}: 5 stories"
        for story, _, tasks in stories:
            assert len(tasks) == 5, f"{story[:40]}: 5 tasks"


def write_markdown(path: Path) -> None:
    total = sum(t[1] for _, _, stories in BACKLOG for _, _, tasks in stories for t in tasks)
    lines = ["# Workshop backlog: AI memory architectures, 2 days x 2 groups", "",
             f"4 epics, 20 user stories, 100 tasks, {total:g} hours estimated. Generated by `tools/build_backlog.py`; "
             "import `backlog.csv` into Azure Boards (hierarchical import).", ""]
    for e_index, (epic, description, stories) in enumerate(BACKLOG, start=1):
        hours = sum(t[1] for _, _, tasks in stories for t in tasks)
        lines += [f"## {epic}", "", description, "", f"Estimate: {hours:g} h", ""]
        for s_index, (story, acceptance, tasks) in enumerate(stories, start=1):
            lines += [f"### US {e_index}.{s_index}", "", story, "", f"**Acceptance:** {acceptance}", "",
                      "| # | Task | Hours | Owner |", "|---|---|---|---|"]
            lines += [f"| {e_index}.{s_index}.{t_index} | {task} | {hours:g} | {owner} |"
                      for t_index, (task, hours, owner) in enumerate(tasks, start=1)]
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Work Item Type", "Title 1", "Title 2", "Title 3", "Description",
                    "Acceptance Criteria", "Original Estimate", "Tags"])
        for e_index, (epic, description, stories) in enumerate(BACKLOG, start=1):
            w.writerow(["", "Epic", epic, "", "", description, "", "", "ai-memory-workshop"])
            for s_index, (story, acceptance, tasks) in enumerate(stories, start=1):
                w.writerow(["", "User Story", "", f"US {e_index}.{s_index} {story[:200]}", "", story, acceptance, "",
                            "ai-memory-workshop"])
                for t_index, (task, hours, owner) in enumerate(tasks, start=1):
                    w.writerow(["", "Task", "", "", f"{e_index}.{s_index}.{t_index} {task}", f"Owner: {owner}", "",
                                hours, f"ai-memory-workshop; {owner}"])


def main() -> int:
    validate()
    (ROOT / "docs").mkdir(exist_ok=True)
    write_markdown(ROOT / "docs" / "backlog.md")
    write_csv(ROOT / "docs" / "backlog.csv")
    print("docs/backlog.md and docs/backlog.csv written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
