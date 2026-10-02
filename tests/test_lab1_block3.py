"""Lab 1 / Block 3: fact memory, Mem0-style consolidation (TODO 1.3-a, 1.3-b)."""

from datetime import date


def _fact(obj, when, conf=0.8, history=None):
    from pdfmem.facts import Fact
    return Fact(subject="scope 2 emissions", predicate="value_percent", object=obj, document="r.pdf", page=1,
                valid_from=when, confidence=conf, history=history or [])


def test_1_3_a_add_when_new():
    from pdfmem.facts import consolidate
    op, fact = consolidate(None, _fact("12 percent", "2025-03-14"))
    assert op == "ADD" and fact.object == "12 percent"


def test_1_3_a_noop_same_value_keeps_best_confidence():
    from pdfmem.facts import consolidate
    op, fact = consolidate(_fact("12 percent", "2025-03-14", 0.6), _fact("12  Percent", "2025-03-14", 0.9))
    assert op == "NOOP" and fact.confidence == 0.9


def test_1_3_a_update_newer_value_keeps_history():
    from pdfmem.facts import consolidate
    op, fact = consolidate(_fact("12 percent", "2025-03-14", history=["9 percent"]), _fact("18 percent", "2026-03-12"))
    assert op == "UPDATE" and fact.object == "18 percent" and fact.history == ["9 percent", "12 percent"]


def test_1_3_a_stale_document_does_not_overwrite():
    from pdfmem.facts import consolidate
    op, fact = consolidate(_fact("18 percent", "2026-03-12"), _fact("12 percent", "2025-03-14"))
    assert op == "NOOP" and fact.object == "18 percent"
    op, _ = consolidate(_fact("18 percent", "2026-03-12"), _fact("5 percent", None))
    assert op == "NOOP"


def test_1_3_rules_extract_table_and_quantities():
    from pdfmem.facts import extract_rules
    from pdfmem.memory import MemoryItem
    item = MemoryItem(id="x", document="cat.pdf", page_start=1, page_end=1, created_at=str(date(2024, 5, 2)),
                      text="|Part No.|Description|Fits|\n|---|---|---|\n|88-2214|Caliper bolt kit|LT-300|\n\n"
                           "Re-torque the caliper bolts to 125 Nm.")
    facts = {(f.subject, f.predicate, f.object) for f in extract_rules(item)}
    assert ("88-2214", "fits", "LT-300") in facts
    assert any(p == "value_nm" and o == "125 Nm" for _, p, o in facts)


def test_1_3_b_recall_by_entity(loaded_chunks):
    from pdfmem.facts import Fact, FactMemory
    memory = FactMemory()
    for f in [Fact(subject="88-2214", predicate="fits", object="LT-300", document="cat.pdf", page=1),
              Fact(subject="74200001", predicate="description", object="Cylinder head gasket", document="cat.pdf", page=1),
              Fact(subject="23V083", predicate="affects", object="LT300", document="recall.pdf", page=1)]:
        assert memory.add(f) == "ADD"
    found = memory.recall("What does part 88-2214 fit?")
    assert [i.kind for i in found] == ["fact"] and "88-2214" in found[0].text
    assert any("Cylinder" in i.text for i in memory.recall("cylinder head gasket description"))
