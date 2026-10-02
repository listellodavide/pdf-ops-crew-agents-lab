"""The PDF ops agent used in Lab 2: answers questions about the inbox with citations.

One step = recall (tool) -> answer. Every step is logged as events, which Lab 2 compresses into
observations (Block 1), turns into episodes and lessons (Block 2) and routes through the hybrid
memory (Block 3). With Foundry configured the answer is written by the model from the recalled
context; offline, the most relevant sentence is extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pdfmem.llm import get_llm
from pdfmem.memory import MemoryItem
from pdfmem.text import QUANTITY, content_words, entities, estimate_tokens, sentences

ANSWER_PROMPT = """Answer the question about the company's PDF documents using only the context.
Quote numbers with their units exactly. End with the citation in brackets, e.g. [file.pdf p.2].
If the context does not contain the answer, say "not found in the documents"."""


@dataclass
class Answer:
    text: str
    citations: list[str]
    items: list[MemoryItem]
    events: list[dict] = field(default_factory=list)
    context_tokens: int = 0


def extractive_answer(question: str, items: list[MemoryItem]) -> tuple[str, str | None]:
    """Offline: the sentence sharing most words and entities with the question (numbers preferred)."""
    q_words, q_ents = set(content_words(question)), entities(question)
    best, best_score, cite = "not found in the documents", 0.0, None
    for rank, item in enumerate(items):
        for s in sentences(item.text):
            score = len(q_words & set(content_words(s))) + len(q_ents & entities(s))
            score += 0.5 * any(ch.isdigit() for ch in s) + 2.0 * bool(QUANTITY.search(s)) - 0.1 * rank
            if score > best_score:
                best, best_score, cite = s, score, item.citation
    return best, cite


class PdfOpsAgent:
    def __init__(self, recall, llm=None, k: int = 5):
        """recall: callable(question, **params) -> list[MemoryItem] (semantic, facts or hybrid)."""
        self.recall, self.llm, self.k = recall, llm or get_llm(), k

    def ask(self, question: str, **params) -> Answer:
        events = [{"kind": "user", "name": "question", "content": question}]
        items = self.recall(question, **params)[: self.k]
        context = "\n\n".join(f"[{i.citation}]\n{i.text}" for i in items)
        events.append({"kind": "tool", "name": "recall", "content": context,
                       "citations": [i.citation for i in items]})
        if self.llm.online:
            text = self.llm.text(ANSWER_PROMPT, f"Question: {question}\n\nContext:\n{context}")
            cites = [i.citation for i in items if i.citation and i.citation.split(" p.")[0] in text]
        else:
            sentence, cite = extractive_answer(question, items)
            text, cites = (f"{sentence} [{cite}]" if cite else sentence), ([cite] if cite else [])
        events.append({"kind": "assistant", "name": "answer", "content": text})
        return Answer(text, cites or [i.citation for i in items[:1]], items, events, estimate_tokens(context))
