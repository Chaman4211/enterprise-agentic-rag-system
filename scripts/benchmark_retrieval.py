import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer

from src.retrieval import semantic_search
from src.vector_store import get_vector_collection


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REPEATS = 3

QUERIES = [
    "Transformer architecture and self-attention mechanisms",
    "Convolutional neural networks for image classification",
    "Graph neural networks for node classification",
    "Reinforcement learning and policy optimization",
    "Language modeling and word embeddings",
    "Deep neural networks for automatic speech recognition",
    "Generative adversarial networks for image synthesis",
    "Stochastic gradient descent and optimization algorithms",
    "Information retrieval and document ranking methods",
    "Question answering and machine reading comprehension",
]


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def main():
    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    _client, collection = get_vector_collection()

    print("Collection:", collection.name)
    print("Stored chunks:", collection.count())
    print("Distance metric:", collection.metadata.get("hnsw:space"))

    print("\nRunning warm-up query...", flush=True)
    semantic_search(
        "What are neural networks?",
        model,
        collection,
        top_k=5,
    )

    all_latencies = []
    query_reports = []

    for index, query in enumerate(QUERIES, start=1):
        query_latencies = []
        results = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            results = semantic_search(query, model, collection, top_k=5)
            query_latencies.append((time.perf_counter() - start) * 1000)

        all_latencies.extend(query_latencies)

        report = {
            "query": query,
            "latencies_ms": query_latencies,
            "average_latency_ms": statistics.mean(query_latencies),
            "results": results,
        }
        query_reports.append(report)

        print(f"\n===== QUERY {index} =====")
        print("Query:", query)
        print(
            f"Average latency: {report['average_latency_ms']:.2f} ms"
        )

        for rank, result in enumerate(results[:3], start=1):
            snippet = result["text"][:220].replace("\n", " ")
            print(
                f"{rank}. Document={result['metadata'].get('document_id')} "
                f"Distance={result['distance']:.4f} "
                f"Similarity={result['similarity']:.4f}"
            )
            print("   ", snippet)

    summary = {
        "collection": collection.name,
        "stored_chunks": collection.count(),
        "distance_metric": collection.metadata.get("hnsw:space"),
        "model": MODEL_NAME,
        "query_count": len(QUERIES),
        "repeats_per_query": REPEATS,
        "measured_operations": len(all_latencies),
        "average_latency_ms": statistics.mean(all_latencies),
        "median_latency_ms": statistics.median(all_latencies),
        "p95_latency_ms": percentile(all_latencies, 0.95),
        "minimum_latency_ms": min(all_latencies),
        "maximum_latency_ms": max(all_latencies),
    }

    output = {
        "summary": summary,
        "queries": query_reports,
    }

    report_path = ROOT / "data/processed/retrieval_benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== BENCHMARK SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved report:", report_path)


if __name__ == "__main__":
    main()
