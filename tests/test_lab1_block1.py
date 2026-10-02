"""Lab 1 / Block 1: optimize the Parquet snapshot and load it (TODO 1.1-a, 1.1-b, 1.1-c)."""

import numpy as np
import polars as pl


def _frame(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return pl.DataFrame({"id": [f"r{i}" for i in range(len(vectors))], "text": [f"t{i}" for i in range(len(vectors))]}
                        ).with_columns(pl.Series("embedding", vectors.tolist(), dtype=pl.Array(pl.Float32, vectors.shape[1])))


def test_1_1_a_near_duplicates_dropped_in_order():
    from pdfmem.optimize import drop_near_duplicates
    frame = _frame([[1, 0, 0], [0.999, 0.01, 0], [0, 1, 0], [0, 0.999, 0.02], [0, 0, 1]])
    kept = drop_near_duplicates(frame, threshold=0.97)
    assert kept["id"].to_list() == ["r0", "r2", "r4"]


def test_1_1_a_threshold_one_keeps_everything_distinct():
    from pdfmem.optimize import drop_near_duplicates
    frame = _frame([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]])
    assert drop_near_duplicates(frame, threshold=0.9999)["id"].to_list() == ["r0", "r1", "r2"]


def test_1_1_c_documents_have_the_index_contract(snapshot):
    from pdfmem.optimize import to_documents
    frame, _ = snapshot
    docs = to_documents(frame, {"recall-23V083-brake-caliper.pdf": "2023-02-20T00:00:00Z"})
    assert len(docs) == frame.height
    recall = next(d for d in docs if d["document"] == "recall-23V083-brake-caliper.pdf")
    expected = {"id", "text", "document", "source_format", "page_start", "page_end", "created_at", "entities",
                "attributes", "embedding"}
    assert expected <= set(recall)
    assert recall["created_at"] == "2023-02-20T00:00:00Z"
    assert "23V083" in recall["entities"] and recall["entities"] == sorted(recall["entities"])
    assert all(c.isalnum() or c in "-_=" for c in recall["id"])
    assert isinstance(recall["embedding"][0], float) and len(recall["embedding"]) == 512


def test_1_1_b_index_per_variant_with_compression(snapshot, fresh_backend):
    from pdfmem import cloud
    from pdfmem.optimize import create_chunk_index
    name = create_chunk_index(512, "binary")
    assert name == cloud.index_name("chunks", "binary")
    schema = cloud.index_client().get_index(name)
    assert schema.vectorSearch["compressions"][0]["kind"] == "binaryQuantization"


def test_1_1_end_to_end_report(snapshot):
    from pdfmem.optimize import optimize
    frame, _ = snapshot
    kept, report = optimize(frame, min_chars=60, threshold=0.97)
    assert report.input_chunks == frame.height
    assert report.kept == kept.height == report.input_chunks - report.dropped_short - report.dropped_exact - report.dropped_near
