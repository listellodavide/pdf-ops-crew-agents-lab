"""Lab 2 / Block 1: observational memory (TODO 2.1-a, 2.1-b)."""

import pytest

TOOL_OUTPUT = ("Part 573 Safety Recall Report 23V-083. Manufacturer: Example Light Trucks LLC. "
               "Number of potentially involved vehicles: 4,812. "
               "Description of defect: the front brake caliper mounting bolts may have been tightened to 95 Nm "
               "instead of the specified 125 Nm. A loose caliper can reduce braking performance. "
               "Owner notification letters are expected to be mailed on 2023-04-10. ") * 3


def test_2_1_a_tool_output_is_compressed_and_keeps_numbers():
    from pdfmem.observe import Observer
    obs = Observer(max_sentences=2).observe(3, {"kind": "tool", "name": "recall", "content": TOOL_OUTPUT,
                                                "citations": ["recall.pdf p.2"]}, "caliper bolt torque")
    assert obs is not None and obs.t == 3 and obs.source == "recall"
    assert "125 Nm" in obs.text and obs.priority == "medium"
    assert len(TOOL_OUTPUT) / len(obs.text) >= 3


def test_2_1_a_unrelated_output_is_low_priority():
    from pdfmem.observe import Observer
    obs = Observer().observe(1, {"kind": "tool", "name": "recall", "content": "The weather was nice. Lunch was good."},
                             "caliper bolt torque")
    assert obs is not None and obs.priority == "low"


def _log(n):
    from pdfmem.observe import Observation, ObservationLog
    log = ObservationLog(max_tokens=10_000)
    for t in range(n):
        log.add(Observation(t, "recall", f"observation number {t} about caliper torque 125 Nm"))
    return log


def test_2_1_b_context_respects_budget_and_prefers_recent():
    from pdfmem.observe import build_context
    from pdfmem.text import estimate_tokens
    context = build_context("You are the PDF ops agent.", "check caliper torque", _log(30),
                            [f"user: question {i}" for i in range(10)], budget=120)
    assert estimate_tokens(context) <= 120 + 8          # separators and section titles
    assert "observation number 29" in context and "observation number 0 " not in context
    assert context.index("Observations:") < context.index("Recent turns:")
    assert "user: question 9" in context


def test_2_1_b_budget_too_small_raises():
    from pdfmem.observe import build_context
    with pytest.raises(ValueError):
        build_context("x" * 400, "goal", _log(1), [], budget=10)


def test_2_1_foundry_memory_store_roundtrip(fresh_backend):
    from pdfmem.observe import FoundryMemory
    memory = FoundryMemory("pdfmem-test")
    memory.ensure("gpt-4.1-mini", "text-embedding-3-small")
    memory.remember_conversation("s1", [{"role": "user", "content": "I prefer answers with the page number."},
                                        {"role": "assistant", "content": "Noted."}])
    found = memory.search("s1", "page number")
    assert any(m["kind"] == "user_profile" for m in found)
    memory.forget("s1")
    assert memory.search("s1", "page number") == []
