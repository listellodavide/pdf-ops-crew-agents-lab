"""Lab 2 / Block 2: episodic memory with reflection (Hindsight-style retain / recall / reflect),
orchestrated as a LangGraph state graph.

    recall_lessons -> act -> evaluate --success--> retain -> END
                                      \\--failure--> reflect -> retain -> END

An episode is what happened (task, retrieval parameters, citations, outcome). A lesson is what to
do differently next time, stored as procedural memory: in the team's memories index (kind
"lesson") and, when configured, mirrored to the Foundry memory store as a procedural memory.

The outcome comes from the golden set (the trainer's ground truth plays the role of user feedback).

    TODO 2.2-a  reflect(): turn a failed episode into a Lesson
    TODO 2.2-b  apply_lessons(): turn the lessons recalled for a new task into retrieval parameters

Run (twice the same questions: the second pass should improve):
    python -m pdfmem.episodic --passes 2
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import operator
import re
import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from pdfmem import cloud
from pdfmem.memory import now_iso, safe_key
from pdfmem.text import content_words, entities


class Lesson(BaseModel):
    trigger: list[str] = Field(description="content words that identify similar tasks")
    prefer_documents: list[str] = Field(default_factory=list, description="document name patterns, e.g. 'recall-*.pdf'")
    prefer_latest: bool = False
    evidence: str = ""

    def text(self) -> str:
        what = ", ".join(self.prefer_documents) or "any document"
        return f"For tasks about {' '.join(self.trigger)}: look in {what}" + (", newest first" if self.prefer_latest else "")


class Episode(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str
    params: dict = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    answer: str = ""
    success: bool = False
    expected_documents: list[str] = Field(default_factory=list)


def family(document: str) -> str:
    """'sustainability-report-2025.pdf' -> 'sustainability-report-*.pdf' (years and numbers wildcarded)."""
    return re.sub(r"\d+", "*", document)


def reflect(episode: Episode) -> Lesson | None:
    """Failed episode -> lesson. Successful episode -> None.

    trigger           the 4 most specific content words of the task (longest first, ties alphabetical)
    prefer_documents  the families of the expected documents (see family())
    prefer_latest     True when the task asks for the latest/most recent/current value
    evidence          the episode id
    """
    if episode.success:
        return None
    # TODO 2.2-a: build and return the Lesson described above.
    # >>> solution
    words = sorted(set(content_words(episode.task)), key=lambda w: (-len(w), w))[:4]
    latest = bool(re.search(r"\b(latest|most recent|current|newest|this year)\b", episode.task.lower()))
    return Lesson(trigger=words, prefer_documents=sorted({family(d) for d in episode.expected_documents}),
                  prefer_latest=latest, evidence=episode.id)
    # <<< solution


def lesson_matches(lesson: Lesson, task: str, min_overlap: int = 2) -> bool:
    return len(set(lesson.trigger) & set(content_words(task))) >= min(min_overlap, len(lesson.trigger))


def apply_lessons(task: str, lessons: list[Lesson], known_documents: list[str]) -> dict:
    """Retrieval parameters for SemanticMemory.recall from the lessons that match the task:
        {"documents": [...known documents matching any preferred pattern...]}   (only if not empty)
        when a matching lesson has prefer_latest, keep only the newest document of each family
        (newest = the highest number in the name, e.g. -2025 beats -2024)."""
    # TODO 2.2-b: implement. Use lesson_matches() and fnmatch.fnmatch(name.lower(), pattern.lower()).
    # >>> solution
    matching = [lesson for lesson in lessons if lesson_matches(lesson, task)]
    patterns = [p for lesson in matching for p in lesson.prefer_documents]
    docs = sorted({d for d in known_documents if any(fnmatch.fnmatch(d.lower(), p.lower()) for p in patterns)})
    if any(lesson.prefer_latest for lesson in matching):
        newest: dict[str, str] = {}
        for d in docs:
            fam = family(d)
            if fam not in newest or re.findall(r"\d+", d) > re.findall(r"\d+", newest[fam]):
                newest[fam] = d
        docs = sorted(newest.values())
    return {"documents": docs} if docs else {}
    # <<< solution


# ----------------------------------------------------------------------- resident storage

class EpisodicMemory:
    """Episodes and lessons in <team>-memories; lessons optionally mirrored to Foundry."""

    def __init__(self, embed=None, foundry=None, scope: str | None = None):
        self.client = cloud.search_client(cloud.index_name("memories"))
        self.embed, self.foundry, self.scope = embed, foundry, scope or f"team-{cloud.team()}"

    def retain_episode(self, episode: Episode) -> None:
        self._put("episode", episode.id, f"{episode.task} -> {'success' if episode.success else 'failure'}",
                  episode.model_dump_json(), entities(episode.task))

    def retain_lesson(self, lesson: Lesson) -> None:
        self._put("lesson", "|".join(lesson.trigger), lesson.text(), lesson.model_dump_json(), set())
        if self.foundry:
            self.foundry.add(self.scope, lesson.text(), kind="procedural")

    def lessons(self) -> list[Lesson]:
        results = self.client.search(search_text="*", filter="kind eq 'lesson'", top=200)
        return [Lesson(**json.loads(r["attributes"])) for r in results]

    def _put(self, kind: str, key: str, text: str, attributes: str, ents: set[str]) -> None:
        doc = {"id": safe_key(f"{kind}|{key}"), "kind": kind, "text": text, "created_at": now_iso(),
               "entities": sorted(ents), "attributes": attributes}
        if self.embed:
            doc["embedding"] = self.embed(text)
        self.client.merge_or_upload_documents([doc])


# --------------------------------------------------------------------------- LangGraph

class State(TypedDict):
    task: str
    expected: list[str]
    params: dict
    answer: str
    citations: list[str]
    success: bool
    lesson: dict | None
    trace: Annotated[list[str], operator.add]


def build_graph(agent, memory: EpisodicMemory, known_documents: list[str]):
    from langgraph.graph import END, START, StateGraph

    def recall_lessons(state: State) -> dict:
        params = apply_lessons(state["task"], memory.lessons(), known_documents)
        return {"params": params, "trace": [f"recall_lessons -> {params or 'none'}"]}

    def act(state: State) -> dict:
        answer = agent.ask(state["task"], **state["params"])
        return {"answer": answer.text, "citations": answer.citations, "trace": [f"act -> {answer.citations[:1]}"]}

    def evaluate(state: State) -> dict:
        cited = {c.split(" p.")[0].lower() for c in state["citations"]}
        ok = bool(cited & {e.lower() for e in state["expected"]})
        return {"success": ok, "trace": [f"evaluate -> {'success' if ok else 'failure'}"]}

    def do_reflect(state: State) -> dict:
        episode = Episode(task=state["task"], params=state["params"], citations=state["citations"],
                          answer=state["answer"], success=state["success"], expected_documents=state["expected"])
        lesson = reflect(episode)
        if lesson:
            memory.retain_lesson(lesson)
        return {"lesson": lesson.model_dump() if lesson else None,
                "trace": [f"reflect -> {lesson.text() if lesson else 'nothing to learn'}"]}

    def retain(state: State) -> dict:
        memory.retain_episode(Episode(task=state["task"], params=state["params"], citations=state["citations"],
                                      answer=state["answer"], success=state["success"],
                                      expected_documents=state["expected"]))
        return {"trace": ["retain"]}

    graph = StateGraph(State)
    for name, fn in [("recall_lessons", recall_lessons), ("act", act), ("evaluate", evaluate),
                     ("reflect", do_reflect), ("retain", retain)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "recall_lessons")
    graph.add_edge("recall_lessons", "act")
    graph.add_edge("act", "evaluate")
    graph.add_conditional_edges("evaluate", lambda s: "retain" if s["success"] else "reflect",
                                {"retain": "retain", "reflect": "reflect"})
    graph.add_edge("reflect", "retain")
    graph.add_edge("retain", END)
    return graph.compile()


def run_pass(app, golden: list[dict]) -> tuple[float, list[dict]]:
    rows = []
    for q in golden:
        out = app.invoke({"task": q["question"], "expected": q["expected_documents"], "params": {}, "answer": "",
                          "citations": [], "success": False, "lesson": None, "trace": []})
        rows.append({"id": q["id"], "success": out["success"], "trace": out["trace"]})
    return sum(r["success"] for r in rows) / max(len(rows), 1), rows


def main() -> int:
    from pdfmem.agent import PdfOpsAgent
    from pdfmem.evaluate import load_golden
    from pdfmem.semantic import SemanticMemory
    p = argparse.ArgumentParser(description="Lab 2 Block 2: episodic memory and reflection.")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--mode", default="vector", help="recall mode of the agent (vector shows the effect best)")
    p.add_argument("-k", type=int, default=1, help="chunks the agent reads per question")
    args = p.parse_args()
    semantic = SemanticMemory()
    semantic.ensure_memories_index()
    agent = PdfOpsAgent(lambda q, **kw: semantic.recall(q, k=args.k, mode=args.mode, **kw), k=args.k)
    known = sorted({r["document"] for r in semantic.client.search(search_text="*", top=1000, select=["document"])})
    app = build_graph(agent, EpisodicMemory(embed=semantic.embed), known)
    golden = load_golden()
    for n in range(1, args.passes + 1):
        rate, rows = run_pass(app, golden)
        print(f"[pass {n}] success {rate:.0%}")
        for r in rows:
            print(f"   {r['id']}: {' | '.join(r['trace'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
