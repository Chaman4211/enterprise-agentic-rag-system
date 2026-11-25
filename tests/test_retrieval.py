import numpy as np
import pytest

from src.retrieval import semantic_search


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.array([[0.6, 0.8]], dtype=np.float32)


class FakeCollection:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Relevant text", "Other text"]],
            "metadatas": [[{"document_id": "paper-1"}, {}]],
            "distances": [[0.1, 0.4]],
        }


def test_semantic_search_returns_ranked_results():
    model = FakeModel()
    collection = FakeCollection()

    results = semantic_search("machine learning", model, collection)

    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["text"] == "Relevant text"
    assert results[0]["metadata"]["document_id"] == "paper-1"
    assert results[0]["distance"] == pytest.approx(0.1)
    assert results[0]["similarity"] == pytest.approx(0.9)
    assert results[1]["similarity"] == pytest.approx(0.6)


def test_query_embedding_is_normalized():
    model = FakeModel()
    collection = FakeCollection()

    semantic_search("machine learning", model, collection)

    assert model.calls[0][1]["normalize_embeddings"] is True
    assert model.calls[0][1]["convert_to_numpy"] is True


def test_top_k_is_passed_to_chromadb():
    model = FakeModel()
    collection = FakeCollection()

    semantic_search("machine learning", model, collection, top_k=3)

    assert collection.calls[0]["n_results"] == 3


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_raises_error(query):
    with pytest.raises(ValueError):
        semantic_search(query, FakeModel(), FakeCollection())


def test_invalid_top_k_raises_error():
    with pytest.raises(ValueError):
        semantic_search("machine learning", FakeModel(), FakeCollection(), 0)


def test_empty_collection_returns_empty_results():
    class EmptyCollection:
        def query(self, **kwargs):
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    results = semantic_search("machine learning", FakeModel(), EmptyCollection())

    assert results == []
