# Labs: memory architectures for PDF agents

Two labs, six blocks, run like a hackathon. Teams of 3 to 4, one laptop drives, everybody reads
the code. Every block is **90 minutes: Part A 40 min, break 10 min, Part B 40 min**.

| | Lab 1: from Parquet to resident memory (Day 1) | Lab 2: agents that remember (Day 2) |
|---|---|---|
| Block 1 | [Optimize the snapshot and load it](lab1_resident_memory/block1_optimize_and_load.md) | [Observation log vs Foundry memory store](lab2_agents_that_remember/block1_observational_memory.md) |
| Block 2 | [Recall modes: keyword, vector, hybrid, semantic, fresh](lab1_resident_memory/block2_recall_modes.md) | [Episodic memory and reflection in LangGraph](lab2_agents_that_remember/block2_episodic_reflection.md) |
| Block 3 | [Fact memory, Mem0-style](lab1_resident_memory/block3_fact_memory.md) | [Graph memory, hybrid layer, Foundry agent](lab2_agents_that_remember/block3_graph_hybrid_agent.md) |

Each block builds on the previous one, and each one ends with a row in your team's scoreboard
(`scoreboard/<team>.csv`) and three lines in your pros/cons notes. At the end of each day the teams
compare architectures with their own numbers, not with vendor claims.

## Rules of the game

1. Part A: implement the TODOs in `pdfmem/` (2 or 3 per block, 10 to 30 lines each) until the
   block's tests pass: `pytest tests/test_lab1_block1.py`.
2. Part B: run the block on the real corpus in the cloud, measure, try one stretch idea, write
   the scoreboard row and your notes.
3. Stuck for more than 10 minutes? `python tools/checkpoint.py 1.2` restores the reference for
   that block (yours is kept as `pdfmem/semantic.py.mine`). No shame: the next block needs it.
4. Points: tests green in Part A (2), scoreboard row (1), best hit@3 of the day per block (1),
   a pros/cons insight the trainer did not expect (1).

## Schedule (same plan for both groups: Group 1 on days 1-2, Group 2 on days 3-4)

| Time | Day 1 (Lab 1) | Day 2 (Lab 2) |
|---|---|---|
| 09:00-09:45 | Theory: memory types, RAG as a base, Parquet snapshot vs resident memory | Theory: short-term vs long-term, episodic, procedural, graph, frameworks |
| 09:45-10:00 | Setup check (`python tools/setup_check.py`) | Recap of Day 1 scoreboards |
| 10:00-11:30 | Block 1.1 (40 + 10 + 40) | Block 2.1 |
| 11:30-11:45 | Break | Break |
| 11:45-13:15 | Block 1.2 | Block 2.2 |
| 13:15-14:00 | Lunch | Lunch |
| 14:00-15:30 | Block 1.3 | Block 2.3 |
| 15:30-15:45 | Break | Break |
| 15:45-17:00 | Scoreboard and pros/cons review | Architecture review, Adobe use cases, PoC backlog |

## Pros/cons notes

Copy `labs/notes-template.md` to `scoreboard/<team>-notes.md` and fill one section per block.
