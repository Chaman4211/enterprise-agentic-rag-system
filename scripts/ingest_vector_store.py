import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_text
from src.vector_store import (
    DEFAULT_COLLECTION_NAME,
    get_vector_collection,
    memory_usage,
)

DATASET = ROOT / "data/processed/clean_corpus.parquet"
CHECKPOINT = ROOT / "data/processed/chroma_ingestion_progress.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 128
PARQUET_BATCH_SIZE = 8


def save_checkpoint(processed_documents: int, ingested_chunks: int) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "collection": DEFAULT_COLLECTION_NAME,
        "dataset": str(DATASET.relative_to(ROOT)),
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "processed_documents": processed_documents,
        "ingested_chunks": ingested_chunks,
    }

    temporary = CHECKPOINT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(CHECKPOINT)


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT.exists():
        return {
            "processed_documents": 0,
            "ingested_chunks": 0,
        }

    state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))

    expected = {
        "collection": DEFAULT_COLLECTION_NAME,
        "dataset": str(DATASET.relative_to(ROOT)),
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: "
                f"{state.get(key)!r} != {value!r}"
            )

    return state


def chunk_id(document_id: str, index: int) -> str:
    value = f"{document_id}:chunk:{index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_metadata(record: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "document_id": str(record["id"]),
        "chunk_index": int(chunk["chunk_index"]),
        "token_count": int(chunk["token_count"]),
    }

    for key in ("source", "created", "added", "language", "license", "url"):
        value = record.get(key)
        if value is not None:
            metadata[key] = str(value)

    metadata_json = record.get("metadata_json")
    if metadata_json:
        metadata["metadata_json"] = str(metadata_json)

    return metadata


def ingest_document(record, tokenizer, model, collection):
    chunks = chunk_text(
        record["text"],
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    document_chunks = 0

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        chunk_batch = chunks[start:start + UPSERT_BATCH_SIZE]
        texts = [chunk["text"] for chunk in chunk_batch]

        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        ids = [
            chunk_id(str(record["id"]), chunk["chunk_index"])
            for chunk in chunk_batch
        ]

        metadatas = [
            build_metadata(record, chunk)
            for chunk in chunk_batch
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            metadatas=metadatas,
        )

        document_chunks += len(chunk_batch)

    return document_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally ingest document chunks into ChromaDB."
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum total documents to process, useful for a pilot run.",
    )
    args = parser.parse_args()

    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be greater than zero")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print("Loading embedding model on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    tokenizer = model.tokenizer

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if CHUNK_SIZE + special_tokens > model.max_seq_length:
        raise ValueError("Chunk size exceeds the model input limit")

    _client, collection = get_vector_collection()

    state = load_checkpoint()
    processed_documents = int(state["processed_documents"])
    ingested_chunks = int(state["ingested_chunks"])

    current_count = collection.count()

    if current_count < ingested_chunks:
        raise RuntimeError(
            "ChromaDB contains fewer chunks than the checkpoint: "
            f"collection={current_count}, checkpoint={ingested_chunks}."
        )

    if current_count > ingested_chunks:
        print(
            "Uncheckpointed chunks detected. "
            "The next document will be safely reprocessed using deterministic IDs."
        )

    print("\n===== INGESTION CONFIGURATION =====")
    print("Collection:", DEFAULT_COLLECTION_NAME)
    print("Distance metric:", collection.metadata.get("hnsw:space"))
    print("Model:", MODEL_NAME)
    print("Embedding dimension:", model.get_embedding_dimension())
    print("Maximum sequence length:", model.max_seq_length)
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Embedding batch size:", EMBEDDING_BATCH_SIZE)
    print("Upsert batch size:", UPSERT_BATCH_SIZE)
    print("Previously completed documents:", processed_documents)
    print("Previously ingested chunks:", ingested_chunks)
    print("Memory before ingestion:", memory_usage())

    parquet = pq.ParquetFile(DATASET)
    total_documents = parquet.metadata.num_rows

    if processed_documents > total_documents:
        raise RuntimeError("Checkpoint exceeds dataset document count")

    if args.max_documents is not None:
        target_documents = min(args.max_documents, total_documents)
        if processed_documents >= target_documents:
            print("Requested document limit is already complete.")
            return
    else:
        target_documents = total_documents

    started = time.perf_counter()
    resume_documents = processed_documents
    document_index = 0

    for batch in parquet.iter_batches(
        columns=[
            "id", "text", "source", "created", "added",
            "language", "license", "url", "metadata_json",
        ],
        batch_size=PARQUET_BATCH_SIZE,
    ):
        for record in batch.to_pylist():
            if document_index < resume_documents:
                document_index += 1
                continue

            if processed_documents >= target_documents:
                break

            if not record.get("id") or not record.get("text"):
                raise ValueError(
                    f"Invalid document at position {processed_documents}"
                )

            added = ingest_document(record, tokenizer, model, collection)

            processed_documents += 1
            document_index += 1
            ingested_chunks += added

            save_checkpoint(processed_documents, ingested_chunks)

            now = time.perf_counter()
            if processed_documents % 10 == 0 or processed_documents == target_documents:
                elapsed = now - started
                print(
                    f"Documents: {processed_documents}/{target_documents} | "
                    f"Chunks: {ingested_chunks:,} | "
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"Memory: {memory_usage().get('VmRSS', 'unknown')}"
                )

        if processed_documents >= target_documents:
            break

    elapsed = time.perf_counter() - started

    actual_count = collection.count()
    if actual_count != ingested_chunks:
        raise RuntimeError(
            f"Final collection count mismatch: "
            f"collection={actual_count}, checkpoint={ingested_chunks}"
        )

    print("\n===== INGESTION RESULT =====")
    print("Documents completed:", processed_documents)
    print("Chunks ingested:", ingested_chunks)
    print("ChromaDB collection count:", actual_count)
    print(f"Elapsed minutes: {elapsed / 60:.2f}")
    print("Memory after ingestion:", memory_usage())
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
