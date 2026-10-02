"""Lab 2 / Block 3, Part B: one memory, four agent frameworks.

The hybrid memory is framework-neutral: it is a function `recall(question) -> cited snippets`.
Each framework only needs a thin adapter:

    Microsoft Foundry Agent Service   pdfmem.hybrid.run_foundry_agent (FunctionTool + MemorySearchPreviewTool)
    LangGraph                         pdfmem.episodic.build_graph (memory nodes in a state graph)
    Semantic Kernel                   PdfMemoryPlugin below (kernel_function)
    AutoGen                           autogen_tool() below (FunctionTool for an AssistantAgent)

Install the optional frameworks with:  pip install -r requirements-frameworks.txt
"""

from __future__ import annotations

import json


def recall_json(memory, question: str, max_items: int = 5) -> str:
    items = memory.recall(question)[:max_items]
    return json.dumps([{"text": i.text[:800], "cite": i.citation, "kind": i.kind} for i in items], ensure_ascii=False)


class PdfMemoryPlugin:
    """Semantic Kernel plugin. kernel.add_plugin(PdfMemoryPlugin(memory), plugin_name="pdf_memory")."""

    def __init__(self, memory):
        self.memory = memory

    try:
        from semantic_kernel.functions import kernel_function

        @kernel_function(name="recall_memory",
                         description="Search the company's PDF memory (chunks, facts, graph, lessons). Returns cited snippets.")
        def recall_memory(self, question: str) -> str:
            return recall_json(self.memory, question)
    except ImportError:                                  # Semantic Kernel not installed
        def recall_memory(self, question: str) -> str:
            return recall_json(self.memory, question)


def autogen_tool(memory):
    """AutoGen FunctionTool; pass it to AssistantAgent(tools=[autogen_tool(memory)], model_client=...)."""
    from autogen_core.tools import FunctionTool

    def recall_memory(question: str) -> str:
        """Search the company's PDF memory (chunks, facts, graph, lessons). Returns cited snippets."""
        return recall_json(memory, question)

    return FunctionTool(recall_memory, description="Search the company's PDF memory and return cited snippets.",
                        name="recall_memory")


SEMANTIC_KERNEL_EXAMPLE = '''
# Semantic Kernel with a Foundry (Azure OpenAI) deployment, Entra ID only
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

token = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
kernel = Kernel()
kernel.add_service(AzureChatCompletion(deployment_name=DEPLOYMENT, endpoint=AOAI_ENDPOINT, ad_token_provider=token))
kernel.add_plugin(PdfMemoryPlugin(memory), plugin_name="pdf_memory")
settings = kernel.get_prompt_execution_settings_from_service_id(None)
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
'''
