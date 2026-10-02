"""pdfmem: memory architectures for PDF agents (Phase 3 of the ingestion pipeline).

Reference implementations used as checkpoints by the labs:

    semantic   Lab 1 Block 1  semantic memory over the Parquet store (recall, remember)
    facts      Lab 1 Block 2  structured fact memory, Mem0-style extraction and consolidation
    multidim   Lab 1 Block 3  multi-dimensional recall (semantic, keyword, time, entity)
    observe    Lab 2 Block 1  observational memory: compressed observation log, context budget
    episodic   Lab 2 Block 2  episodic memory with reflection (retain, recall, reflect)
    graph      Lab 2 Block 3  knowledge graph memory, multi-hop queries
    hybrid     Lab 2 Block 3  the hybrid memory layer: router, governance, audit
"""

__version__ = "1.0.0"
