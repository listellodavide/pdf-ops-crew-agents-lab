# Lab 2, Block 1: observational memory vs the Foundry memory store (90 min)

**Memory type:** short-term / working memory of an agent session.
Two ways to keep a long session small:

| | Our observation log (Mastra-style) | Foundry memory store (preview) |
|---|---|---|
| What is kept | one dated line per event, with citations | chat summary, user profile, procedural memories |
| Who decides | your Observer and Reflector code | the service's extraction model |
| Control | full: budget, priorities, format | configuration only (kinds, TTL) |
| Cost | none offline, or one small model call per tool output | model + embedding calls in the service |
| Where it lives | in the agent's context | resident in Foundry, shared across sessions and agents |

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 2.1-a | `pdfmem/observe.py` `Observer.observe` (offline path) | Keep the most salient sentences of a tool output, with its citation |
| 2.1-b | `pdfmem/observe.py` `build_context` | Pack system, goal, newest observations and turns within a token budget |

Done when `pytest tests/test_lab2_block1.py` is green.

## Break (10 min)

## Part B (40 min): run a session and compare

```powershell
python -m pdfmem.observe --budget 400
python -m pdfmem.observe --budget 400 --foundry
```

1. Write down the compression ratio (raw tokens / observation tokens). The literature reports 3-6x
   for text and up to 40x for tool outputs: what do you get on PDF recall outputs?
2. Lower `--budget` to 150. Which observations survive the Reflector? Is anything important lost?
3. Read what the Foundry memory store extracted (`memory/lab2_block1.json`). What did it keep that
   your log did not, and the reverse?
4. Stretch: let the model write the observations (unset `PDFMEM_OFFLINE`). Compare quality,
   latency and tokens with the rule-based Observer.

## Notes
Observation logs are cheap and predictable but lossy. Managed memory is convenient and resident,
but you do not control what it extracts, and it is a preview service: plan a fallback.
