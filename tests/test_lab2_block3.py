"""Lab 2 / Block 3: graph memory and the hybrid layer (TODO 2.3-a, 2.3-b)."""

import pytest


@pytest.mark.parametrize("question,expected", [
    ("Which part do I need for the vehicles affected by recall 23V083?", ["graph", "facts", "semantic"]),
    ("What is the latest scope 1 and 2 emissions reduction?", ["facts", "fresh"]),
    ("What did we learn last time about caliper questions?", ["episodic", "semantic"]),
    ("Which vehicles does 23V083 cover?", ["facts", "semantic"]),
    ("How is tire tread depth checked?", ["semantic"]),
])
def test_2_3_b_route(question, expected):
    from pdfmem.hybrid import route
    assert route(question) == expected


def _graph():
    from pdfmem.graph import GraphMemory
    g = GraphMemory()
    g.add_edge("23V-083", "affects", "LT-300", "recall.pdf", 1)
    g.add_edge("88-2214", "fits", "LT-300", "catalog.pdf", 1)
    g.add_edge("74200001", "fits", "FH16", "catalog.pdf", 1)
    g.add_edge("LT-300", "mentioned_with", "D13", "manual.pdf", 2)
    return g


def test_2_3_a_two_hop_recall_to_part(loaded_chunks):
    paths = _graph().two_hop("23V083")
    ends = {(first[2], second[1], second[2]) for first, second in paths}
    assert ("LT300", "~fits", "882214") in ends
    assert all(second[2] != "23V083" for _, second in paths)
    assert not any(second[1] == "~mentioned_with" or second[1] == "mentioned_with" for _, second in paths)


def test_2_3_a_two_hop_with_comentions(loaded_chunks):
    ends = {second[2] for _, second in _graph().two_hop("23V083", include_comentions=True)}
    assert {"882214", "D13"} <= ends


def test_2_3_governance_forget_and_redact(loaded_chunks):
    from pdfmem import cloud
    from pdfmem.hybrid import HybridMemory, redact
    from pdfmem.semantic import SemanticMemory
    memory = HybridMemory(SemanticMemory("scalar", meta=loaded_chunks))
    removed = memory.forget("owner-manual-lt300.pdf")
    assert removed[cloud.index_name("chunks", "scalar")] >= 1
    left = cloud.search_client(cloud.index_name("chunks", "scalar")).search(
        search_text="*", filter="document eq 'owner-manual-lt300.pdf'")
    assert list(left) == []
    audit = list(memory.memories.search(search_text="*", filter="kind eq 'audit'"))
    assert audit and "forget" in audit[0]["text"]
    assert redact("mail ana.pop@adobe.com or +40 721 123 456") == "mail <email> or <phone>"
