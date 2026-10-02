"""Lab 2 / Block 3a: knowledge graph memory for multi-hop questions.

Edges live in the team's memories index (kind "edge"): subject -predicate-> object, with the
document and page that support them. Two sources:
    facts whose subject and object are both entities   (88-2214 -fits-> LT-300, 23V083 -affects-> LT-300)
    entities mentioned in the same chunk                (co-mention, weaker)

"Which part do I need for the vehicles in recall 23V083?" is two hops:
    23V083 -affects-> LT300 <-fits- 882214

A dedicated graph database (Cosmos DB for Gremlin, Neo4j) is the right tool when traversals go
deeper than 2-3 hops or the graph is large; here the edges sit next to the other memories and the
traversal runs in Python, which is enough to compare the architecture.

    TODO 2.3-a  two_hop(): paths of length two from an entity
"""

from __future__ import annotations

import argparse
import itertools

from pdfmem import cloud
from pdfmem.memory import MemoryItem, now_iso, safe_key
from pdfmem.text import entities, normalize_entity

Edge = tuple[str, str, str, str]          # (source, predicate, target, citation)


class GraphMemory:
    def __init__(self):
        self.client = cloud.search_client(cloud.index_name("memories"))

    def add_edge(self, source: str, predicate: str, target: str, document: str | None, page: int | None) -> None:
        s, t = normalize_entity(source), normalize_entity(target)
        if not s or not t or s == t:
            return
        self.client.merge_or_upload_documents([{
            "id": safe_key(f"edge|{s}|{predicate}|{t}"), "kind": "edge", "text": f"{s} {predicate} {t}",
            "subject": s, "predicate": predicate, "object": t, "document": document, "page": page,
            "entities": sorted({s, t}), "created_at": now_iso()}])

    def build_from_facts(self) -> int:
        added = 0
        for r in self.client.search(search_text="*", filter="kind eq 'fact'", top=5000):
            subject_ents, object_ents = entities(r["subject"] or ""), entities(r["object"] or "")
            if subject_ents and object_ents:
                for s, t in itertools.product(subject_ents, object_ents):
                    self.add_edge(s, r["predicate"], t, r.get("document"), r.get("page"))
                    added += 1
        return added

    def build_from_chunks(self, chunks: list[MemoryItem], max_entities: int = 5) -> int:
        added = 0
        for chunk in chunks:
            ents = sorted(entities(chunk.text))[:max_entities]
            for a, b in itertools.combinations(ents, 2):
                self.add_edge(a, "mentioned_with", b, chunk.document, chunk.page_start)
                added += 1
        return added

    def neighbors(self, entity: str, include_comentions: bool = True) -> list[Edge]:
        """Edges touching the entity, as (entity, predicate, other, citation), in both directions."""
        e = normalize_entity(entity)
        flt = f"kind eq 'edge' and (subject eq '{e}' or object eq '{e}')"
        if not include_comentions:
            flt += " and predicate ne 'mentioned_with'"
        out = []
        for r in self.client.search(search_text="*", filter=flt, top=500):
            other = r["object"] if r["subject"] == e else r["subject"]
            direction = r["predicate"] if r["subject"] == e else f"~{r['predicate']}"   # ~ = reverse edge
            out.append((e, direction, other, f"{r.get('document')} p.{r.get('page')}"))
        return out

    def two_hop(self, entity: str, include_comentions: bool = False) -> list[tuple[Edge, Edge]]:
        """All paths start -> middle -> end with end != start, as pairs of edges from neighbors()."""
        # TODO 2.3-a: for each edge of neighbors(entity), take neighbors(middle) and keep the second
        # edges whose end is not the start entity. Pass include_comentions through. Avoid duplicates.
        raise NotImplementedError("TODO 2.3-a: see the comment above")

    def mermaid(self, entity: str) -> str:
        lines = ["graph LR"]
        for first, second in self.two_hop(entity):
            for s, p, t, _ in (first, second):
                a, b = (t, s) if p.startswith("~") else (s, t)
                lines.append(f"  {a} -- {p.lstrip('~')} --> {b}")
        return "\n".join(dict.fromkeys(lines))


def main() -> int:
    p = argparse.ArgumentParser(description="Lab 2 Block 3: graph memory.")
    p.add_argument("entity", help="e.g. 23V083")
    p.add_argument("--build", action="store_true", help="(re)build edges from the stored facts first")
    args = p.parse_args()
    from pdfmem.semantic import SemanticMemory
    SemanticMemory().ensure_memories_index()
    graph = GraphMemory()
    if args.build:
        print(f"[graph] {graph.build_from_facts()} edges from facts")
    for first, second in graph.two_hop(args.entity):
        print(f"{first[0]} -{first[1]}-> {first[2]} -{second[1]}-> {second[2]}   [{first[3]}; {second[3]}]")
    print(graph.mermaid(args.entity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
