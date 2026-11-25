from typing import Any


def semantic_search(
    query: str,
    model: Any,
    collection: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the most similar chunks for a natural-language query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    response = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results = []

    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        results.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
        )

    return results
