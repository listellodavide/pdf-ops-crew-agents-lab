"""Lab 2 / Block 2: episodic memory and reflection in LangGraph (TODO 2.2-a, 2.2-b)."""

from types import SimpleNamespace

KNOWN = ["owner-manual-lt300.pdf", "recall-23V083-brake-caliper.pdf", "sustainability-report-2024.pdf",
         "sustainability-report-2025.pdf", "truck-parts-catalog-engine.pdf"]


def test_2_2_a_reflect_failed_episode():
    from pdfmem.episodic import Episode, reflect
    lesson = reflect(Episode(id="e1", task="What is the most recent scope 1 and 2 emissions reduction?",
                             expected_documents=["sustainability-report-2025.pdf"], success=False))
    assert lesson.prefer_documents == ["sustainability-report-*.pdf"]
    assert lesson.prefer_latest is True and lesson.evidence == "e1"
    assert lesson.trigger == ["emissions", "reduction", "scope"]


def test_2_2_a_success_has_nothing_to_learn():
    from pdfmem.episodic import Episode, reflect
    assert reflect(Episode(task="anything", success=True)) is None


def test_2_2_b_apply_lessons_newest_document():
    from pdfmem.episodic import Lesson, apply_lessons
    lessons = [Lesson(trigger=["emissions", "reduction", "scope"], prefer_documents=["sustainability-report-*.pdf"],
                      prefer_latest=True)]
    assert apply_lessons("latest scope 2 emissions reduction", lessons, KNOWN) == {"documents": ["sustainability-report-2025.pdf"]}
    assert apply_lessons("tire pressure front axle", lessons, KNOWN) == {}


def test_2_2_b_without_latest_keeps_the_family():
    from pdfmem.episodic import Lesson, apply_lessons
    lessons = [Lesson(trigger=["steel", "tonnes"], prefer_documents=["sustainability-report-*.pdf"])]
    assert apply_lessons("tonnes of steel recovered", lessons, KNOWN) == {
        "documents": ["sustainability-report-2024.pdf", "sustainability-report-2025.pdf"]}


def test_2_2_graph_learns_between_passes(loaded_chunks, golden):
    from pdfmem.episodic import EpisodicMemory, build_graph, run_pass

    class OldReportAgent:          # always cites the 2024 report unless lessons restrict documents
        def ask(self, question, documents=None):
            doc = (documents or ["sustainability-report-2024.pdf"])[0]
            return SimpleNamespace(text="...", citations=[f"{doc} p.1"])

    recent = [q for q in golden if q["type"] == "recent"]
    app = build_graph(OldReportAgent(), EpisodicMemory(), KNOWN)
    first, _ = run_pass(app, recent)
    second, rows = run_pass(app, recent)
    assert first == 0.0 and second == 1.0, rows
