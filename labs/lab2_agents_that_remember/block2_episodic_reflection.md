# Lab 2, Block 2: episodic memory and reflection in LangGraph (90 min)

**Memory types:** episodic (what happened) and procedural (what to do next time), Hindsight-style
retain / recall / reflect. **Where they live:** `pdfmem-<team>-memories` (kinds `episode`,
`lesson`); lessons can be mirrored to the Foundry memory store as procedural memories.

```
recall_lessons -> act -> evaluate --success--> retain -> END
                                   \--failure--> reflect -> retain -> END
```

The agent answers the golden questions; the golden set plays the user who says "wrong document".
After pass 1 the agent should do better in pass 2 **without any code or prompt change**.

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 2.2-a | `pdfmem/episodic.py` `reflect` | A failed episode becomes a lesson: trigger words, document families, "newest first" |
| 2.2-b | `pdfmem/episodic.py` `apply_lessons` | Lessons that match a new task become retrieval parameters |

Done when `pytest tests/test_lab2_block2.py` is green.

## Break (10 min)

## Part B (40 min): learn from experience

```powershell
python -m pdfmem.episodic --passes 2 --mode vector -k 1
python -m pdfmem.episodic --passes 2 --mode hybrid -k 3
```

1. Success rate of pass 1 vs pass 2. Which lessons were created? Read them in the index
   (kind `lesson`) or in the Foundry portal if you mirrored them.
2. Over-learning: ask a question that shares trigger words with a lesson but needs another
   document. Does the lesson hurt? How would you limit a lesson (expiry, confidence, counter-examples)?
3. Stretch: draw the graph (`app.get_graph().draw_mermaid()`) and add a human-in-the-loop node that
   asks for approval before a lesson is stored (LangGraph `interrupt`).

## Notes
Episodic memory turns feedback into behaviour without retraining. The risk is learning the wrong
thing from one bad episode: lessons need evidence, expiry and an audit trail.
