"""Lab 1 / Block 3: structured fact memory (Mem0-style) resident in the team's memories index.

Chunks are long and repetitive; a fact is one line with provenance:

    (88-2214, fits, LT-300)                       truck-parts-catalog.pdf p.1
    (scope 2 emissions, value_percent, 18 percent)  sustainability-report-2025.pdf p.1   valid_from 2026-03-12

Pipeline: extract (Foundry structured output, or rules offline) -> consolidate against what is
already stored (ADD / UPDATE / NOOP, TODO 1.3-a) -> upsert with history -> recall by entity (TODO 1.3-b).

Run:
    python -m pdfmem.facts extract --documents "sustainability-report-2025.pdf"
    python -m pdfmem.facts ask "most recent scope 1 and 2 emissions reduction"
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from pdfmem import cloud, schema
from pdfmem.llm import get_llm
from pdfmem.memory import MemoryItem, now_iso, safe_key
from pdfmem.text import QUANTITY, STOPWORDS, entities, normalize_entity, sentences

Operation = Literal["ADD", "UPDATE", "NOOP"]
VERBS = set("fell rose decreased increased reduced recovered processed affects affected is are was were "
            "be been may have has had should must shall will tightened mailed expected".split())


class Fact(BaseModel):
    subject: str = Field(description="the thing the fact is about, e.g. a part number or a metric")
    predicate: str = Field(description="relation or attribute, snake_case, e.g. fits, torque, value_percent")
    object: str = Field(description="the value, with unit if any")
    document: str | None = None
    page: int | None = None
    valid_from: str | None = Field(default=None, description="ISO date the source document was issued")
    confidence: float = 0.8
    history: list[str] = Field(default_factory=list, description="previous values, oldest first")

    @property
    def key(self) -> str:
        return f"{normalize_entity(self.subject) or self.subject.lower()}|{self.predicate}"

    def as_text(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ')} {self.object}"


class FactList(BaseModel):
    facts: list[Fact]


EXTRACTION_PROMPT = """Extract durable facts from the PDF excerpt: specifications, part numbers and what they fit,
measured values with units, affected vehicles, dates. One fact per value. Use short subjects
(part number, model code, metric name), snake_case predicates and keep units in the object.
Do not invent facts that are not in the text."""


# --------------------------------------------------------------------------------- extraction

def _subject_before(sentence: str, start: int) -> str:
    """The noun phrase before a quantity: '... caliper mounting bolts may have been tightened to' -> 'caliper mounting bolts'."""
    words = re.findall(r"[\w-]+", sentence[:start])
    while words and (words[-1].lower() in STOPWORDS or words[-1].lower() in VERBS):
        words.pop()
    phrase = []
    for w in reversed(words):
        if w.lower() in STOPWORDS or w.lower() in VERBS or len(phrase) == 4:
            break
        phrase.insert(0, w)
    return " ".join(phrase).lower()


def extract_rules(item: MemoryItem) -> list[Fact]:
    """Offline extractor: Markdown table rows and 'value + unit' sentences."""
    facts: list[Fact] = []
    page = item.page_start
    lines = item.text.splitlines()
    header: list[str] | None = None
    for line in lines:                                          # | Part No. | Description | Qty | Fits |
        if line.startswith("|---"):                             # separator row: still the same table
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = [re.sub(r"\W+", "_", c.lower()).strip("_") for c in cells]
                continue
            subject = cells[0]
            for name, value in zip(header[1:], cells[1:], strict=False):
                if value:
                    facts.append(Fact(subject=subject, predicate=name, object=value, document=item.document,
                                      page=page, valid_from=item.created_at, confidence=0.95))
        else:
            header = None
    for sentence in sentences(item.text):
        if sentence.startswith("|"):
            continue
        for m in QUANTITY.finditer(sentence):
            subject = _subject_before(sentence, m.start())
            if subject:
                unit = m.group("unit").lower().replace("%", "percent").replace(" ", "_")
                facts.append(Fact(subject=subject, predicate=f"value_{unit}", object=m.group(0), document=item.document,
                                  page=page, valid_from=item.created_at, confidence=0.6))
        ents = entities(sentence)
        campaigns = [e for e in ents if re.fullmatch(r"\d{2}V\d{3,6}", e)]
        models = [e for e in ents if e not in campaigns and not e.isdigit()]
        for c in campaigns:
            for mdl in models:
                facts.append(Fact(subject=c, predicate="affects", object=mdl, document=item.document, page=page,
                                  valid_from=item.created_at, confidence=0.7))
    return facts


def extract_facts(item: MemoryItem, llm=None) -> list[Fact]:
    llm = llm or get_llm()
    if not llm.online:
        return extract_rules(item)
    found = llm.parse(EXTRACTION_PROMPT, f"Document: {item.citation}\n\n{item.text}", FactList).facts
    return [f.model_copy(update={"document": item.document, "page": f.page or item.page_start,
                                 "valid_from": item.created_at}) for f in found]


# ------------------------------------------------------------------------------ consolidation

def consolidate(existing: Fact | None, new: Fact) -> tuple[Operation, Fact]:
    """Decide what to store when `new` arrives and `existing` has the same key (subject + predicate).

        no existing fact                      -> ADD    new
        same object (case/space-insensitive)  -> NOOP   existing, keeping the higher confidence
        different object, new is newer or     -> UPDATE new, with history = existing.history + [existing.object]
          has no date while existing has none
        different object, new is older        -> NOOP   existing (a stale document must not overwrite)
    """
    # TODO 1.3-a: implement the four cases above. Compare dates as ISO strings (valid_from may be None:
    # treat a missing date as older than any date).
    raise NotImplementedError("TODO 1.3-a: see the comment above")


class FactMemory:
    """Facts resident in <team>-memories (kind = "fact"), one document per subject + predicate."""

    def __init__(self, embed=None, dimensions: int | None = None):
        self.index = cloud.index_name("memories")
        self.embed = embed                          # callable text -> vector (snapshot model)
        if dimensions:
            cloud.index_client().create_or_update_index(schema.memories_index(self.index, dimensions))
        self.client = cloud.search_client(self.index)

    def get(self, key: str) -> Fact | None:
        try:
            doc = self.client.get_document(safe_key("fact|" + key))
        except Exception:
            return None
        return Fact(**json.loads(doc["attributes"]))

    def add(self, fact: Fact) -> Operation:
        op, stored = consolidate(self.get(fact.key), fact)
        if op != "NOOP" or stored.confidence != fact.confidence:
            text = stored.as_text()
            doc = {"id": safe_key("fact|" + fact.key), "kind": "fact", "text": text, "subject": stored.subject,
                   "predicate": stored.predicate, "object": stored.object, "document": stored.document,
                   "page": stored.page, "valid_from": stored.valid_from, "created_at": now_iso(),
                   "entities": sorted(entities(text) | {normalize_entity(stored.subject)} - {""}),
                   "attributes": stored.model_dump_json()}
            if self.embed:
                doc["embedding"] = self.embed(text)
            self.client.merge_or_upload_documents([doc])
        return op

    def recall(self, question: str, k: int = 5) -> list[MemoryItem]:
        """Facts about the entities in the question; falls back to keyword search on fact text."""
        ents = sorted(entities(question))
        # TODO 1.3-b: when the question has entities, search with filter
        #   "kind eq 'fact' and entities/any(e: search.in(e, 'A|B', '|'))"  ... or simpler, one
        #   "entities/any(e: e eq 'A')" clause per entity joined with " or ".
        # Otherwise use search_text=question with filter "kind eq 'fact'". Return MemoryItems (kind="fact").
        raise NotImplementedError("TODO 1.3-b: see the comment above")


def main() -> int:
    from pdfmem.semantic import SemanticMemory
    p = argparse.ArgumentParser(description="Lab 1 Block 3: fact memory.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="extract facts from the chunks of some documents")
    e.add_argument("--documents", nargs="+", required=True)
    e.add_argument("--limit", type=int, default=200, help="max chunks")
    a = sub.add_parser("ask")
    a.add_argument("question")
    args = p.parse_args()
    semantic = SemanticMemory()
    semantic.ensure_memories_index()
    facts = FactMemory(embed=semantic.embed)
    if args.cmd == "extract":
        quoted = " or ".join(f"document eq '{d}'" for d in args.documents)
        chunks = [MemoryItem.from_search(r) for r in semantic.client.search(search_text="*", filter=quoted, top=args.limit)]
        ops: dict[str, int] = {}
        for chunk in sorted(chunks, key=lambda c: (c.created_at or "", c.page_start or 0)):
            for fact in extract_facts(chunk):
                op = facts.add(fact)
                ops[op] = ops.get(op, 0) + 1
        print(f"[facts] {len(chunks)} chunks -> {ops}")
    else:
        for item in facts.recall(args.question):
            print(f"- {item.text}   [{item.citation}] history={item.attributes.get('history')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
