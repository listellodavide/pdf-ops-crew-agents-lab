# Lab 2, Block 3: graph memory, the hybrid layer and a Foundry agent (90 min)

**Memory types:** knowledge graph (entities and relations) and the hybrid layer that routes a
question to the right memories, with governance. **Agent:** a versioned prompt agent in Foundry
Agent Service whose tool is your memory.

```
"Which part do I need for the vehicles in recall 23V083?"
   route -> graph, facts, semantic
   graph:  23V083 -affects-> LT300 <-fits- 882214
```

## Part A (40 min): implement

| TODO | File | What |
|---|---|---|
| 2.3-a | `pdfmem/graph.py` `two_hop` | Paths of length two, without going back to the start |
| 2.3-b | `pdfmem/hybrid.py` `route` | The routing table: graph, facts, fresh, episodic, semantic |

Done when `pytest tests/test_lab2_block3.py` is green.

## Break (10 min)

## Part B (40 min): put it together

```powershell
python -m pdfmem.graph 23V083 --build
python -m pdfmem.hybrid ask "Which part do I need for the vehicles affected by recall 23V083?"
python -m pdfmem.hybrid agent "Which part do I need for the vehicles affected by recall 23V083?"
python -m pdfmem.hybrid agent "What is the latest scope 1 and 2 emissions reduction?" --foundry-memory
python -m pdfmem.hybrid forget nhtsa-tsb-MC-10231577.pdf
```

1. Run five golden questions through `hybrid ask` and through `semantic` only. Where does routing
   help, where does it add tokens without adding answers?
2. Governance: `forget` a document, then check chunks, facts, edges and the audit entry. What is
   still remembered elsewhere (Foundry memory store, the Parquet snapshot)? What would a
   right-to-be-forgotten request need?
3. Frameworks (pick one, `pip install -r requirements-frameworks.txt`): expose the same memory to
   Semantic Kernel (`PdfMemoryPlugin`) or AutoGen (`autogen_tool`) from `pdfmem/frameworks.py`.
   Compare with LangGraph (Block 2) and Foundry Agent Service (this block): what does each
   framework give you, what does it cost you?
4. Final scoreboard: fill the last row and the recommendation section of your notes.

## Notes
A real graph database (Cosmos DB for Gremlin, Neo4j) pays off for deep traversals; here two hops
over edges stored next to the other memories were enough. Write down at which question volume or
depth you would switch.
