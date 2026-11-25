import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = Path("data/processed/clean_corpus.parquet")
OUTPUT = Path("data/processed/token_length_profile.json")
BATCH_SIZE = 128
MODEL_INPUT_LIMIT = 256


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_token_limit = MODEL_INPUT_LIMIT - special_token_count

    if content_token_limit <= 0:
        raise ValueError("Model input limit must exceed special-token count")

    parquet = pq.ParquetFile(DATASET)
    lengths = []
    total = 0

    print("Profiling document token lengths...")

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

        lengths.extend(len(tokens) for tokens in encoded["input_ids"])
        total += len(texts)

        if total % 1000 < BATCH_SIZE:
            print(f"Documents profiled: {total}")

    values = np.asarray(lengths, dtype=np.int64)

    report = {
        "model": MODEL_NAME,
        "dataset": str(DATASET),
        "documents_profiled": int(total),
        "model_input_limit": MODEL_INPUT_LIMIT,
        "special_token_count": special_token_count,
        "content_token_limit": content_token_limit,
        "statistics": {
            "mean": round(float(np.mean(values)), 2),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "maximum": int(np.max(values)),
            "minimum": int(np.min(values)),
        },
        "documents_exceeding_model_limit": int(
            np.sum(values > content_token_limit)
        ),
        "percentage_exceeding_model_limit": round(
            float(np.mean(values > content_token_limit) * 100), 2
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))

    print("\n===== TOKEN LENGTH PROFILE =====")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
