"""Azure AI Search index definitions for the resident memory.

chunks index    one per team and compression variant; semantic memory over the PDF snapshot
memories index  one per team; facts, graph edges, episodes, lessons, notes, audit ("kind" field)
"""

from __future__ import annotations

from datetime import timedelta

from azure.search.documents.indexes.models import (
    BinaryQuantizationCompression,
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    HnswAlgorithmConfiguration,
    RescoringOptions,
    ScalarQuantizationCompression,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    TagScoringFunction,
    TagScoringParameters,
    VectorSearch,
    VectorSearchProfile,
)

T = SearchFieldDataType
SEMANTIC_CONFIG = "default"
SCORING_PROFILE = "fresh-entities"
ENTITY_TAGS_PARAMETER = "ents"
VARIANTS = ("none", "scalar", "binary")


def compression_for(variant: str):
    """Vector compression for a variant. Rescoring keeps the full-precision originals so the
    top candidates found on compressed vectors are re-ranked exactly (oversampling 4x)."""
    rescoring = RescoringOptions(enable_rescoring=True, default_oversampling=4.0,
                                 rescore_storage_method="preserveOriginals")
    if variant == "scalar":
        return ScalarQuantizationCompression(compression_name="sq-int8", rescoring_options=rescoring)
    if variant == "binary":
        return BinaryQuantizationCompression(compression_name="bq-1bit", rescoring_options=rescoring)
    if variant == "none":
        return None
    raise ValueError(f"unknown variant {variant!r}; use one of {VARIANTS}")


def vector_search(compression) -> VectorSearch:
    return VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vec", algorithm_configuration_name="hnsw",
                                      compression_name=compression.compression_name if compression else None)],
        compressions=[compression] if compression else [],
    )


def scoring_profiles() -> list[ScoringProfile]:
    """Server-side multi-dimensional ranking: newer documents and matching entities score higher."""
    return [ScoringProfile(name=SCORING_PROFILE, functions=[
        FreshnessScoringFunction(field_name="created_at", boost=2.0, interpolation="linear",
                                 parameters=FreshnessScoringParameters(boosting_duration=timedelta(days=730))),
        TagScoringFunction(field_name="entities", boost=3.0,
                           parameters=TagScoringParameters(tags_parameter=ENTITY_TAGS_PARAMETER)),
    ])]


def chunk_index(name: str, dimensions: int, variant: str = "scalar") -> SearchIndex:
    compression = compression_for(variant)
    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=T.String, key=True),
            SearchableField(name="text"),
            SimpleField(name="document", type=T.String, filterable=True, facetable=True),
            SimpleField(name="source_format", type=T.String, filterable=True, facetable=True),
            SimpleField(name="page_start", type=T.Int32, filterable=True, sortable=True),
            SimpleField(name="page_end", type=T.Int32, filterable=True),
            SimpleField(name="created_at", type=T.DateTimeOffset, filterable=True, sortable=True),
            SearchField(name="entities", type=T.Collection(T.String), filterable=True, searchable=True),
            SimpleField(name="attributes", type=T.String),
            # stored=False: the vector is searchable but never returned, which saves storage
            SearchField(name="embedding", type=T.Collection(T.Single), searchable=True, stored=False,
                        vector_search_dimensions=dimensions, vector_search_profile_name="vec"),
        ],
        vector_search=vector_search(compression),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG, prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="text")]))]),
        scoring_profiles=scoring_profiles(),
    )


def memories_index(name: str, dimensions: int) -> SearchIndex:
    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=T.String, key=True),
            SimpleField(name="kind", type=T.String, filterable=True, facetable=True),   # fact|edge|episode|lesson|note|audit
            SearchableField(name="text"),
            SearchableField(name="subject", filterable=True),
            SimpleField(name="predicate", type=T.String, filterable=True, facetable=True),
            SearchableField(name="object", filterable=True),
            SimpleField(name="document", type=T.String, filterable=True, facetable=True),
            SimpleField(name="page", type=T.Int32, filterable=True),
            SimpleField(name="valid_from", type=T.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="created_at", type=T.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="expires_at", type=T.DateTimeOffset, filterable=True),
            SearchField(name="entities", type=T.Collection(T.String), filterable=True, searchable=True),
            SimpleField(name="attributes", type=T.String),
            SearchField(name="embedding", type=T.Collection(T.Single), searchable=True, stored=False,
                        vector_search_dimensions=dimensions, vector_search_profile_name="vec"),
        ],
        vector_search=vector_search(compression_for("scalar")),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG, prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="text")]))]),
    )
