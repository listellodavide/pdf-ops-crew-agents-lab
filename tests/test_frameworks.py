"""Lab 2 / Block 3, Part B: the hybrid memory behind Semantic Kernel and AutoGen (optional installs)."""

import asyncio
import json

import pytest


@pytest.fixture()
def hybrid(loaded_chunks):
    """Any object with recall(question) -> MemoryItems; here a plain keyword query (no TODO needed)."""
    from pdfmem import cloud
    from pdfmem.memory import MemoryItem

    class KeywordMemory:
        def recall(self, question, k=5):
            client = cloud.search_client(cloud.index_name("chunks", "scalar"))
            return [MemoryItem.from_search(r) for r in client.search(search_text=question, top=k)]

    return KeywordMemory()


def test_semantic_kernel_plugin(hybrid):
    pytest.importorskip("semantic_kernel")
    from semantic_kernel import Kernel

    from pdfmem.frameworks import PdfMemoryPlugin
    kernel = Kernel()
    kernel.add_plugin(PdfMemoryPlugin(hybrid), plugin_name="pdf_memory")
    result = asyncio.run(kernel.invoke(plugin_name="pdf_memory", function_name="recall_memory",
                                       question="cold tire pressure"))
    items = json.loads(str(result))
    assert items and items[0]["cite"].startswith("owner-manual-lt300.pdf")


def test_autogen_tool(hybrid):
    pytest.importorskip("autogen_core")
    from autogen_core import CancellationToken

    from pdfmem.frameworks import autogen_tool
    tool = autogen_tool(hybrid)
    assert tool.name == "recall_memory"
    out = asyncio.run(tool.run_json({"question": "caliper bolts torque"}, CancellationToken()))
    assert "recall-23V083-brake-caliper.pdf" in tool.return_value_as_string(out)
