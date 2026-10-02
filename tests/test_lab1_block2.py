"""Lab 1 / Block 2: recall modes on the resident semantic memory (TODO 1.2-a, 1.2-b)."""

import pytest


def test_1_2_b_documents_filter_syntax():
    from pdfmem.semantic import documents_filter
    assert documents_filter([]) is None
    assert documents_filter(["a.pdf", "b, c.pdf"]) == "search.in(document, 'a.pdf|b, c.pdf', '|')"
    assert documents_filter(["o'brien.pdf"]) == "search.in(document, 'o''brien.pdf', '|')"


@pytest.mark.parametrize("mode,text,vector,semantic,profile", [
    ("keyword", True, False, False, False), ("vector", False, True, False, False),
    ("hybrid", True, True, False, False), ("semantic", True, True, True, False), ("fresh", True, True, False, True)])
def test_1_2_a_search_kwargs_per_mode(mode, text, vector, semantic, profile):
    from pdfmem.semantic import search_kwargs
    kw = search_kwargs(mode, "caliper bolts LT-300", [0.1, 0.2], 5, "document eq 'x'")
    assert kw["top"] == 5 and kw["filter"] == "document eq 'x'" and "text" in kw["select"]
    assert bool(kw.get("search_text")) is text
    assert bool(kw.get("vector_queries")) is vector
    assert (kw.get("query_type") == "semantic") is semantic
    assert (kw.get("scoring_profile") is not None) is profile
    if profile:
        assert kw["scoring_parameters"] == ["ents-LT300"]


def test_1_2_recall_finds_the_recall_report(loaded_chunks):
    from pdfmem.semantic import SemanticMemory
    memory = SemanticMemory("scalar", meta=loaded_chunks)
    items = memory.recall("front brake caliper bolts torque 23V083", k=3, mode="hybrid")
    assert items[0].document == "recall-23V083-brake-caliper.pdf"
    only_manual = memory.recall("tire pressure", k=3, mode="keyword", documents=["owner-manual-lt300.pdf"])
    assert {i.document for i in only_manual} == {"owner-manual-lt300.pdf"}


def test_1_2_fresh_prefers_the_newest_report(loaded_chunks):
    from pdfmem.semantic import SemanticMemory
    memory = SemanticMemory("scalar", meta=loaded_chunks)
    top = memory.recall("scope 1 and scope 2 emissions fell", k=2, mode="fresh")[0]
    assert top.document == "sustainability-report-2025.pdf"
