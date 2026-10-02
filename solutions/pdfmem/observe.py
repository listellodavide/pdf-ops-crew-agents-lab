"""Lab 2 / Block 1: observational memory (short-term) vs the Foundry managed memory store.

Long agent sessions over PDFs fill the context with tool outputs: each recall returns pages of
text. Observational memory keeps a compressed log instead of the raw history:

    Observer   one event -> one short, dated observation with its source        TODO 2.1-a
    Reflector  merges and drops observations when the log exceeds its budget (given)
    Context    system + goal + newest observations + last turns, within a token budget  TODO 2.1-b

Then the same session goes to the Foundry memory store (chat summary + user profile), which does
the extraction in the service. Compare tokens, control and what each one remembers.

Run:
    python -m pdfmem.observe --budget 400
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from pdfmem import cloud, config
from pdfmem.llm import get_llm
from pdfmem.text import QUANTITY, content_words, entities, estimate_tokens, sentences

PRIORITY = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Observation:
    t: int
    source: str
    text: str
    priority: str = "medium"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.line())

    def line(self) -> str:
        return f"[t{self.t}] {self.source}: {self.text}"


def salience(sentence: str, goal: str) -> float:
    """How useful a sentence is for the goal: shared words, shared entities, numbers with units."""
    return (len(set(content_words(goal)) & set(content_words(sentence)))
            + 2 * len(entities(goal) & entities(sentence))
            + 1.5 * bool(QUANTITY.search(sentence)))


class Observer:
    def __init__(self, llm=None, max_sentences: int = 2):
        self.llm, self.max_sentences = llm or get_llm(), max_sentences

    def observe(self, t: int, event: dict, goal: str) -> Observation | None:
        """Compress one event. user -> keep the question; tool -> the most salient sentences with
        their citation; assistant -> the answer, high priority. Returns None for empty events."""
        content = (event.get("content") or "").strip()
        if not content:
            return None
        if event["kind"] == "user":
            return Observation(t, "user", content[:200], "high")
        if event["kind"] == "assistant":
            return Observation(t, "agent", content[:240], "high")
        if self.llm.online:
            text = self.llm.text("Compress this tool output into at most two short sentences that keep every "
                                 "number, unit, part number and citation relevant to the goal.",
                                 f"Goal: {goal}\n\nTool output:\n{content}")
            return Observation(t, event.get("name", "tool"), text.strip(), "medium")
        # TODO 2.1-a (offline observer): rank sentences(content) by salience(sentence, goal), keep the
        # best self.max_sentences in their original order, join them with " ", and return an
        # Observation(t, event.get("name", "tool"), joined_text, priority) where priority is "medium"
        # if the best salience is > 1 else "low". Sentences starting with "[" (citations) or "|" are skipped.
        # >>> solution
        candidates = [s for s in sentences(content) if not s.startswith(("[", "|"))]
        if not candidates:
            return None
        ranked = sorted(range(len(candidates)), key=lambda i: -salience(candidates[i], goal))
        chosen = sorted(ranked[: self.max_sentences])
        best = salience(candidates[ranked[0]], goal)
        cites = ", ".join(event.get("citations", [])[:2])
        text = " ".join(candidates[i] for i in chosen) + (f" ({cites})" if cites else "")
        return Observation(t, event.get("name", "tool"), text, "medium" if best > 1 else "low")
        # <<< solution


class ObservationLog:
    def __init__(self, max_tokens: int = 600):
        self.items: list[Observation] = []
        self.max_tokens = max_tokens

    def add(self, observation: Observation | None) -> None:
        if observation:
            self.items.append(observation)
            if self.tokens > self.max_tokens:
                self.reflect()

    @property
    def tokens(self) -> int:
        return sum(o.tokens for o in self.items)

    def reflect(self) -> None:
        """Reflector: drop exact repeats, then the oldest low-priority observations, until within budget."""
        seen, unique = set(), []
        for o in self.items:
            key = re.sub(r"\W+", " ", o.text.lower())
            if key not in seen:
                seen.add(key)
                unique.append(o)
        self.items = unique
        while self.tokens > self.max_tokens and len(self.items) > 1:
            victim = min(self.items, key=lambda o: (PRIORITY[o.priority], o.t))
            self.items.remove(victim)


def build_context(system: str, goal: str, log: ObservationLog, recent_turns: list[str], budget: int) -> str:
    """Pack the prompt in this priority order, never exceeding `budget` tokens:
    1. system, 2. goal, 3. recent turns (newest first, kept in chronological order),
    4. observations (newest first, kept in chronological order). Raise ValueError if 1+2 do not fit."""
    # TODO 2.1-b: implement. Use estimate_tokens(text) for every piece; separate pieces with "\n".
    # Output layout: system, "Goal: " + goal, "Observations:" + the kept observation lines,
    # "Recent turns:" + the kept turns.
    # >>> solution
    head = [system, f"Goal: {goal}"]
    used = sum(estimate_tokens(p) for p in head)
    if used > budget:
        raise ValueError("system prompt and goal do not fit in the budget")
    turns: list[str] = []
    for turn in reversed(recent_turns):
        cost = estimate_tokens(turn)
        if used + cost > budget:
            break
        turns.insert(0, turn)
        used += cost
    lines: list[str] = []
    for o in reversed(log.items):
        cost = estimate_tokens(o.line())
        if used + cost > budget:
            break
        lines.insert(0, o.line())
        used += cost
    return "\n".join(head + ["Observations:", *lines, "Recent turns:", *turns])
    # <<< solution


def compression_ratio(events: list[dict], log: ObservationLog) -> float:
    raw = sum(estimate_tokens(e.get("content") or "") for e in events)
    return raw / max(log.tokens, 1)


# ------------------------------------------------------------------- Foundry managed memory

class FoundryMemory:
    """Thin wrapper over project.beta.memory_stores (preview). One store per team; scope = session/user."""

    def __init__(self, name: str | None = None):
        self.name = name or f"pdfmem-{cloud.team()}"
        self.stores = cloud.project().beta.memory_stores

    def ensure(self, chat_model: str, embedding_model: str, ttl_days: int | None = 30) -> None:
        from azure.ai.projects.models import MemoryStoreDefaultDefinition, MemoryStoreDefaultOptions
        try:
            self.stores.get(self.name)
            return
        except Exception:
            pass
        from datetime import timedelta
        options = MemoryStoreDefaultOptions(user_profile_enabled=True, chat_summary_enabled=True,
                                            procedural_memory_enabled=True,
                                            default_ttl_seconds=timedelta(days=ttl_days) if ttl_days else None)
        self.stores.create(name=self.name, description="Lab 2 resident memory",
                           definition=MemoryStoreDefaultDefinition(chat_model=chat_model,
                                                                  embedding_model=embedding_model, options=options))

    def remember_conversation(self, scope: str, messages: list[dict]):
        """The service extracts user profile / chat summary / procedural memories (asynchronous)."""
        poller = self.stores.begin_update_memories(self.name, scope=scope, items=messages, update_delay=0)
        return poller.result()

    def add(self, scope: str, content: str, kind: str = "procedural"):
        return self.stores.create_memory(self.name, scope=scope, content=content, kind=kind)

    def search(self, scope: str, query: str, max_memories: int = 5) -> list[dict]:
        from azure.ai.projects.models import MemorySearchOptions
        result = self.stores.search_memories(self.name, scope=scope, items=query,
                                             options=MemorySearchOptions(max_memories=max_memories))
        return [{"kind": str(m.memory_item.kind), "content": m.memory_item.content} for m in result.memories]

    def forget(self, scope: str) -> None:
        self.stores.delete_scope(self.name, scope=scope)


def main() -> int:
    from pdfmem.agent import PdfOpsAgent
    from pdfmem.evaluate import load_golden
    from pdfmem.semantic import SemanticMemory
    p = argparse.ArgumentParser(description="Lab 2 Block 1: observation log vs Foundry memory store.")
    p.add_argument("--budget", type=int, default=400, help="context budget in tokens")
    p.add_argument("--foundry", action="store_true", help="also send the session to the Foundry memory store")
    args = p.parse_args()
    semantic = SemanticMemory()
    agent, observer, log = PdfOpsAgent(semantic.recall), Observer(), ObservationLog(max_tokens=args.budget)
    events, turns = [], []
    for q in load_golden():
        answer = agent.ask(q["question"])
        for event in answer.events:
            events.append(event)
            log.add(observer.observe(len(events), event, q["question"]))
        turns += [f"user: {q['question']}", f"agent: {answer.text}"]
    context = build_context("You are the PDF ops agent.", "answer follow-up questions", log, turns[-4:], args.budget)
    raw = sum(estimate_tokens(e["content"]) for e in events)
    print(f"[observe] {len(events)} events, {raw} raw tokens -> {log.tokens} observation tokens "
          f"(ratio {compression_ratio(events, log):.1f}x); context {estimate_tokens(context)} / {args.budget}")
    print(context)
    result = {"raw_tokens": raw, "log_tokens": log.tokens, "ratio": round(compression_ratio(events, log), 2)}
    if args.foundry:
        import os
        memory = FoundryMemory()
        memory.ensure(os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
                      os.environ.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"))
        messages = [{"role": "user" if t.startswith("user:") else "assistant", "content": t.split(": ", 1)[1]}
                    for t in turns]
        update = memory.remember_conversation("session-1", messages)
        found = memory.search("session-1", "torque and emissions")
        print(f"\n[foundry] {len(update.memory_operations)} memory operations; search ->")
        for m in found:
            print(f"  {m['kind']}: {m['content'][:160]}")
        result["foundry_memories"] = found
    out = config.MEMORY_DIR / "lab2_block1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
