"""Lab 2 / Block 3b: the hybrid memory layer, its governance, and a Foundry agent that uses it.

    router      question -> which memories to ask                                  TODO 2.3-b
    recall      ask them, merge with provenance, stay within a token budget
    governance  forget(document) everywhere, expire, redact before writing, audit trail
    agent       a Foundry prompt agent with recall_memory as a function tool (+ optional
                Foundry memory search tool), answering from the resident memory

Run:
    python -m pdfmem.hybrid ask "Which part do I need for the vehicles in recall 23V083?"
    python -m pdfmem.hybrid agent "Which part do I need for the vehicles in recall 23V083?"
    python -m pdfmem.hybrid forget "owner-manual-lt300.pdf"
"""

from __future__ import annotations

import argparse
import json
import os
import re

from pdfmem import cloud
from pdfmem.memory import MemoryItem, now_iso, safe_key
from pdfmem.text import entities, estimate_tokens

ROUTES = ("graph", "facts", "semantic", "fresh", "episodic")
MULTI_HOP = re.compile(r"\b(which|what)\b.*\b(part|parts|fit|fits|need)\b.*\b(recall|campaign|affected)\b"
                       r"|\b(affected by|related to|linked to)\b", re.IGNORECASE)
TEMPORAL = re.compile(r"\b(latest|most recent|current|newest|this year|last year)\b", re.IGNORECASE)
EXPERIENCE = re.compile(r"\b(last time|previously|before|lesson|learned|again)\b", re.IGNORECASE)
PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
       (re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"), "<iban>"),
       (re.compile(r"\+?\d[\d\s/-]{8,}\d"), "<phone>")]


def route(question: str) -> list[str]:
    """Which memories to ask, in order of trust:
        multi-hop question (MULTI_HOP)      -> ["graph", "facts", "semantic"]
        temporal question (TEMPORAL)        -> ["facts", "fresh"]
        about past experience (EXPERIENCE)  -> ["episodic", "semantic"]
        has entities                        -> ["facts", "semantic"]
        otherwise                           -> ["semantic"]
    Check the rules in this order and return the first that applies."""
    # TODO 2.3-b: implement the routing table above.
    raise NotImplementedError("TODO 2.3-b: see the comment above")


def redact(text: str) -> str:
    for pattern, label in PII:
        text = pattern.sub(label, text)
    return text


class HybridMemory:
    def __init__(self, semantic, facts=None, graph=None, episodic=None, budget_tokens: int = 1200):
        self.semantic, self.facts, self.graph, self.episodic = semantic, facts, graph, episodic
        self.budget = budget_tokens
        self.memories = cloud.search_client(cloud.index_name("memories"))

    def recall(self, question: str, k: int = 5, **params) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for source in route(question):
            if source == "graph" and self.graph:
                for ent in entities(question):
                    for first, second in self.graph.two_hop(ent):
                        text = f"{first[0]} {first[1]} {first[2]}; {second[0]} {second[1]} {second[2]}"
                        items.append(MemoryItem(id=f"path:{text}", text=text, kind="graph",
                                                document=first[3].split(" p.")[0], attributes={"via": [first[3], second[3]]}))
            elif source == "facts" and self.facts:
                items += self.facts.recall(question, k)
            elif source == "episodic" and self.episodic:
                items += [MemoryItem(id=f"lesson:{lesson.text()}", text=lesson.text(), kind="lesson")
                          for lesson in self.episodic.lessons()][:k]
            elif source in ("semantic", "fresh"):
                items += self.semantic.recall(question, k=k, mode="fresh" if source == "fresh" else "hybrid", **params)
        packed, used, seen = [], 0, set()
        for item in items:                                  # trust order, then budget
            cost = estimate_tokens(item.text)
            if item.id in seen or used + cost > self.budget:
                continue
            seen.add(item.id)
            packed.append(item)
            used += cost
        return packed

    # ------------------------------------------------------------------------ governance
    def audit(self, op: str, detail: dict) -> None:
        self.memories.merge_or_upload_documents([{
            "id": safe_key(f"audit|{now_iso()}|{op}|{json.dumps(detail, sort_keys=True)}"), "kind": "audit",
            "text": f"{op} {json.dumps(detail)}", "created_at": now_iso(),
            "attributes": json.dumps({"op": op, "actor": os.environ.get("USERNAME") or os.environ.get("USER"), **detail})}])

    def forget(self, document: str) -> dict:
        """Right to be forgotten / withdrawn document: remove it from every resident memory."""
        removed = {}
        for name, client, kind_filter in ((self.semantic.index, self.semantic.client, None),
                                          (cloud.index_name("memories"), self.memories, "kind ne 'audit'")):
            flt = f"document eq '{document.replace(chr(39), chr(39) * 2)}'" + (f" and {kind_filter}" if kind_filter else "")
            ids = [{"id": r["id"]} for r in client.search(search_text="*", filter=flt, top=10000, select=["id"])]
            if ids:
                client.delete_documents(ids)
            removed[name] = len(ids)
        self.audit("forget", {"document": document, "removed": removed})
        return removed

    def expire(self, now: str | None = None) -> int:
        now = now or now_iso()
        ids = [{"id": r["id"]} for r in self.memories.search(search_text="*", filter=f"expires_at lt {now}",
                                                             top=10000, select=["id"])]
        if ids:
            self.memories.delete_documents(ids)
        self.audit("expire", {"removed": len(ids)})
        return len(ids)

    def remember_note(self, text: str, ttl_days: int | None = 30, embed=None) -> str:
        from datetime import UTC, datetime, timedelta
        clean = redact(text)
        doc = {"id": safe_key(f"note|{clean}"), "kind": "note", "text": clean, "created_at": now_iso(),
               "entities": sorted(entities(clean)),
               "expires_at": (datetime.now(UTC) + timedelta(days=ttl_days)).isoformat(timespec="seconds") if ttl_days else None}
        if embed:
            doc["embedding"] = embed(clean)
        self.memories.merge_or_upload_documents([doc])
        self.audit("remember", {"kind": "note", "chars": len(clean)})
        return clean


# ------------------------------------------------------------------------- Foundry agent

RECALL_TOOL = {
    "name": "recall_memory",
    "description": "Search the company's PDF memory (chunks, facts, graph, lessons). Returns cited snippets.",
    "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                   "required": ["question"], "additionalProperties": False},
}


def run_foundry_agent(memory: HybridMemory, question: str, use_foundry_memory: bool = False) -> str:
    """A versioned prompt agent in Foundry Agent Service whose tool is our hybrid memory.
    The tool runs here (function calling); the Foundry memory search tool runs in the service."""
    from azure.ai.projects.models import (
        FunctionTool,
        MemorySearchOptions,
        MemorySearchPreviewTool,
        PromptAgentDefinition,
    )
    project = cloud.project()
    tools = [FunctionTool(strict=True, **RECALL_TOOL)]
    if use_foundry_memory:
        tools.append(MemorySearchPreviewTool(memory_store_name=f"pdfmem-{cloud.team()}", scope=f"team-{cloud.team()}",
                                             search_options=MemorySearchOptions(max_memories=5)))
    agent = project.agents.create_version(agent_name=f"pdfmem-{cloud.team()}-agent", definition=PromptAgentDefinition(
        model=os.environ["MODEL_DEPLOYMENT_NAME"], tools=tools,
        instructions="Answer questions about the PDF inbox. Always call recall_memory first. Cite [document p.N]."))
    client = project.get_openai_client()
    ref = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
    try:
        response = client.responses.create(input=question, extra_body=ref)
        for _ in range(4):                                   # tool loop, bounded
            calls = [o for o in response.output if o.type == "function_call"]
            if not calls:
                break
            outputs = []
            for call in calls:
                args = json.loads(call.arguments)
                items = memory.recall(args["question"])
                outputs.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps([{"text": i.text[:800], "cite": i.citation, "kind": i.kind}
                                                      for i in items])})
            response = client.responses.create(input=outputs, previous_response_id=response.id, extra_body=ref)
        return response.output_text
    finally:
        project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)


def build(dimensions_from_snapshot: bool = True) -> HybridMemory:
    from pdfmem.episodic import EpisodicMemory
    from pdfmem.facts import FactMemory
    from pdfmem.graph import GraphMemory
    from pdfmem.semantic import SemanticMemory
    semantic = SemanticMemory()
    semantic.ensure_memories_index()
    return HybridMemory(semantic, FactMemory(embed=semantic.embed), GraphMemory(), EpisodicMemory(embed=semantic.embed))


def main() -> int:
    p = argparse.ArgumentParser(description="Lab 2 Block 3: hybrid memory layer.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "agent"):
        s = sub.add_parser(name)
        s.add_argument("question")
        s.add_argument("--foundry-memory", action="store_true")
    f = sub.add_parser("forget")
    f.add_argument("document")
    args = p.parse_args()
    memory = build()
    if args.cmd == "ask":
        print(f"route: {route(args.question)}")
        for item in memory.recall(args.question):
            print(f"- [{item.kind}] {item.text[:160]!r}  {item.citation}")
    elif args.cmd == "agent":
        print(run_foundry_agent(memory, args.question, args.foundry_memory))
    else:
        print(memory.forget(args.document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
