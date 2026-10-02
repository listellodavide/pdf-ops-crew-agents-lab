"""Access to the resident memory in the cloud: Microsoft Foundry project and Azure AI Search.

Everything authenticates with Entra ID (az login / DefaultAzureCredential). No keys.

.env
    PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
    MODEL_DEPLOYMENT_NAME=gpt-4.1-mini            chat model for extraction, reflection, answers
    EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small   only for models.json key "foundry-te3s"
    SEARCH_ENDPOINT=https://<search>.search.windows.net   optional: else the project's default
                                                           Azure AI Search connection is used
    PDFMEM_TEAM=g1-t3                             prefixes every index and memory scope per team
    PDFMEM_BACKEND=azure                          azure (default) or simulator (contingency/tests)

Index layout per team (fits an AI Search Basic tier: 2 groups x 4 teams x 2 indexes + trainer):
    <team>-chunks     semantic memory loaded from the optimized Parquet snapshot
    <team>-memories   facts, graph edges, episodes, lessons, notes (field "kind")
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from pdfmem import config  # noqa: F401  (loads .env)


def team() -> str:
    raw = os.environ.get("PDFMEM_TEAM", "demo").lower()
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-") or "demo"


def index_name(kind: str, variant: str | None = None) -> str:
    """AI Search index names: lowercase letters, digits and dashes, max 128 characters."""
    name = f"pdfmem-{team()}-{kind}" + (f"-{variant}" if variant else "")
    return name[:128]


def backend() -> str:
    return os.environ.get("PDFMEM_BACKEND", "azure").lower()


@lru_cache(maxsize=1)
def credential():
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential(exclude_interactive_browser_credential=False)


@lru_cache(maxsize=1)
def project():
    """The Foundry project client (agents, memory stores, connections, OpenAI-compatible client)."""
    if backend() == "simulator":
        from pdfmem.simulator import SimulatedProject
        return SimulatedProject()
    from azure.ai.projects import AIProjectClient
    endpoint = os.environ.get("PROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("PROJECT_ENDPOINT is not set (see .env.example)")
    return AIProjectClient(endpoint=endpoint, credential=credential())


@lru_cache(maxsize=1)
def search_endpoint() -> str:
    """SEARCH_ENDPOINT, or the Azure AI Search resource connected to the Foundry project."""
    if os.environ.get("SEARCH_ENDPOINT"):
        return os.environ["SEARCH_ENDPOINT"]
    from azure.ai.projects.models import ConnectionType
    connection = project().connections.get_default(ConnectionType.AZURE_AI_SEARCH)
    return connection.target


@lru_cache(maxsize=1)
def index_client():
    if backend() == "simulator":
        from pdfmem.simulator import SimulatedIndexClient
        return SimulatedIndexClient()
    from azure.search.documents.indexes import SearchIndexClient
    return SearchIndexClient(endpoint=search_endpoint(), credential=credential())


def search_client(name: str):
    if backend() == "simulator":
        return index_client().get_search_client(name)
    from azure.search.documents import SearchClient
    return SearchClient(endpoint=search_endpoint(), index_name=name, credential=credential())


def reset_clients() -> None:
    """Forget cached clients (tests switch backends)."""
    for f in (project, search_endpoint, index_client, credential):
        f.cache_clear()
