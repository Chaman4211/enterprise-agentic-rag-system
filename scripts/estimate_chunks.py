import math
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = Path("data/processed/clean_corpus.parquet")
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
BATCH_SIZE = 32


def count_chunks(token_count: int) -> int:
    if token_count <= 0:
        return 0

    if token_count <= CHUNK_SIZE:
        return 1

    stride = CHUNK_SIZE - CHUNK_OVERLAP
    return 1 + math.ceil((token_count - CHUNK_SIZE) / stride)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    parquet = pq.ParquetFile(DATASET)

    total_documents = 0
    total_tokens = 0
    total_chunks = 0
    maximum_chunks = 0

    print("Estimating chunks using the actual embedding tokenizer...")

    for batch in parquet.iter_batches(
        columns=["text"],
        batch_size=BATCH_SIZE,
    ):
        texts = batch.column("text").to_pylist()

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        for _, token_ids in zip(texts, encoded["input_ids"]):
            token_count = len(token_ids)
            chunks = count_chunks(token_count)

            total_documents += 1
            total_tokens += token_count
            total_chunks += chunks
            maximum_chunks = max(maximum_chunks, chunks)

        if total_documents % 512 < BATCH_SIZE:
            print(f"Documents processed: {total_documents}")

    print("\n===== CHUNK ESTIMATE =====")
    print(f"Dataset documents: {total_documents:,}")
    print(f"Total document tokens: {total_tokens:,}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Chunk stride: {CHUNK_SIZE - CHUNK_OVERLAP}")
    print(f"Estimated total chunks: {total_chunks:,}")
    print(f"Average chunks per document: {total_chunks / total_documents:.2f}")
    print(f"Maximum chunks in one document: {maximum_chunks:,}")


if __name__ == "__main__":
    main()
