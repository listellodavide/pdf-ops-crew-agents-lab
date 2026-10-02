"""Text utilities shared by every memory type: tokens, sentences, entities, BM25, token estimates."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
will with which what who how when where why does do did can should must may our your their than then
there these those also into over per after before about most latest recent current new newest""".split())

# Domain entities that matter in recalls, catalogs and reports. Keys are normalized so that
# "23V-083" == "23V083" and "LT-300" == "LT300".
ENTITY_PATTERNS = {
    "campaign": re.compile(r"\b\d{2}V-?\d{3}(?:\d{3})?\b"),
    "part_number": re.compile(r"\b\d{2}-\d{4}\b|\b\d{7,10}\b|\b[A-Z]\d{5,}\b"),
    "model": re.compile(r"\b[A-Z]{1,4}-?\d{1,4}[A-Z]?\b"),
    "year": re.compile(r"\b(?:19|20)\d{2}\b"),
}
QUANTITY = re.compile(
    r"(?P<value>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?"
    r"(?P<unit>Nm|bar|km|kWh|percent|%|tonnes|hours|cubic metres|degrees|kg|mm)\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t for t in WORD.findall(text.lower()) if len(t) > 1]


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_END.split(text)]
    return [s for s in parts if len(s) > 3 and not s.startswith("|---")]


def normalize_entity(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def entities(text: str) -> set[str]:
    """Normalized domain entities (campaigns, part numbers, model codes, years)."""
    found: set[str] = set()
    for pattern in ENTITY_PATTERNS.values():
        found.update(normalize_entity(m.group(0)) for m in pattern.finditer(text))
    return {e for e in found if len(e) >= 3}


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English; good enough for budgeting."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def bm25(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    avg = sum(map(len, docs)) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    scores = np.zeros(n, dtype=np.float32)
    for term in set(content_words(query)):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / max(avg, 1)))
    return scores


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high - low < 1e-9 else (values - low) / (high - low)
