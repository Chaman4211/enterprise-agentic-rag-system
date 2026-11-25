import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = Path("data/processed/clean_corpus.parquet")
MAX_CHUNKS = 256
BATCH_SIZE = 16
TOTAL_CHUNKS = 527103


def memory_usage():
    result = {}

    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                result[key] = value.strip()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU embedding throughput.")
    parser.add_argument("--total-chunks", type=int, default=TOTAL_CHUNKS)
    args = parser.parse_args()

    if args.total_chunks <= 0:
        parser.error("--total-chunks must be greater than zero")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print("Loading embedding model on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    tokenizer = model.tokenizer

    print("Collecting a representative chunk sample...")
    parquet = pq.ParquetFile(DATASET)
    sample_chunks = []

    for batch in parquet.iter_batches(
        columns=["id", "text"],
        batch_size=8,
    ):
        for record in batch.to_pylist():
            chunks = chunk_text(
                record["text"],
                tokenizer,
                chunk_size=220,
                chunk_overlap=40,
            )

            for chunk in chunks:
                sample_chunks.append(chunk["text"])

                if len(sample_chunks) >= MAX_CHUNKS:
                    break

            if len(sample_chunks) >= MAX_CHUNKS:
                break

        if len(sample_chunks) >= MAX_CHUNKS:
            break

    if len(sample_chunks) < 16:
        raise RuntimeError("Not enough chunks available for benchmarking")

    print(f"Sample chunks: {len(sample_chunks)}")
    print(f"Embedding batch size: {BATCH_SIZE}")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    print("Memory before embedding:", memory_usage())

    print("\nRunning warm-up...")
    model.encode(
        sample_chunks[:8],
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    measured_chunks = sample_chunks[8:]

    print(f"\nBenchmarking {len(measured_chunks)} chunks...")
    start = time.perf_counter()

    embeddings = model.encode(
        measured_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    elapsed = time.perf_counter() - start
    throughput = len(measured_chunks) / elapsed
    estimated_seconds = args.total_chunks / throughput

    print("\n===== EMBEDDING BENCHMARK =====")
    print(f"Measured chunks: {len(measured_chunks)}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Chunks per second: {throughput:.2f}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Estimated full-corpus embedding time: {estimated_seconds / 3600:.2f} hours")
    print("Memory after embedding:", memory_usage())


if __name__ == "__main__":
    main()
