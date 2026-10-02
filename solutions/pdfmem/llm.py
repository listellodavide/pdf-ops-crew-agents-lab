"""One small interface over Microsoft Foundry (Responses API, Entra ID) or no model at all.

Every lab runs offline: each memory component has a deterministic rule-based path, and uses the
model only when one is configured. Set in .env:

    PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
    MODEL_DEPLOYMENT_NAME=gpt-4.1-mini
    PDFMEM_OFFLINE=1            # force the rule-based path even when Foundry is configured

Authentication is Entra ID only (az login). No API keys.
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from pdfmem import config  # noqa: F401  (loads .env)

T = TypeVar("T", bound=BaseModel)


class OfflineLLM:
    online = False
    name = "offline (rules)"

    def text(self, system: str, user: str) -> str:
        raise RuntimeError("no model configured; callers must use their rule-based path")

    def parse(self, system: str, user: str, schema: type[T]) -> T:
        raise RuntimeError("no model configured; callers must use their rule-based path")


class FoundryLLM:
    online = True

    def __init__(self, endpoint: str, deployment: str):
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        self._project = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
        self._client = self._project.get_openai_client()
        self.deployment = deployment
        self.name = f"foundry:{deployment}"

    def text(self, system: str, user: str) -> str:
        response = self._client.responses.create(model=self.deployment, instructions=system, input=user)
        return response.output_text

    def parse(self, system: str, user: str, schema: type[T]) -> T:
        response = self._client.responses.parse(model=self.deployment, instructions=system, input=user,
                                                text_format=schema)
        return response.output_parsed


_cached = None


def get_llm():
    """Foundry when configured and not forced offline, else the offline stand-in."""
    global _cached
    if _cached is None:
        endpoint, deployment = os.environ.get("PROJECT_ENDPOINT"), os.environ.get("MODEL_DEPLOYMENT_NAME")
        if endpoint and deployment and os.environ.get("PDFMEM_OFFLINE", "0") != "1":
            try:
                _cached = FoundryLLM(endpoint, deployment)
            except ImportError:
                print("[pdfmem] azure-ai-projects not installed: running offline")
                _cached = OfflineLLM()
        else:
            _cached = OfflineLLM()
    return _cached
