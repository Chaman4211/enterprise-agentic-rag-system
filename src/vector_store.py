from pathlib import Path

import chromadb


DEFAULT_COLLECTION_NAME = "agentic_rag_chunks"
DEFAULT_DISTANCE_METRIC = "cosine"


def get_vector_collection(
    persist_directory: str | Path = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Open or create a persistent ChromaDB collection.

    Existing collections must use cosine distance. The function refuses
    to silently reuse a collection configured with another metric.
    """
    client = chromadb.PersistentClient(path=str(persist_directory))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": DEFAULT_DISTANCE_METRIC},
    )

    metric = (collection.metadata or {}).get("hnsw:space")

    if metric != DEFAULT_DISTANCE_METRIC:
        raise ValueError(
            f"Collection {collection_name!r} uses {metric!r}; "
            f"expected {DEFAULT_DISTANCE_METRIC!r}"
        )

    return client, collection


def memory_usage() -> dict[str, str]:
    """Return process resident and peak memory from Linux procfs."""
    values: dict[str, str] = {}

    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                values[key] = value.strip()

    return values
