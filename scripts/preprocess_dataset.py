import gzip
import json
import os
import tempfile
import sys
from collections import Counter
from pathlib import Path

# Add the project root so scripts can import the src package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing import preprocess_record

RAW_FILE = Path("data/raw/arxiv_papers.jsonl.gz")
OUTPUT_FILE = Path("data/processed/clean_corpus.parquet")

BATCH_SIZE = 100

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("source", pa.string()),
    ("created", pa.string()),
    ("added", pa.string()),
    ("language", pa.string()),
    ("text_length", pa.int64()),
    ("license", pa.string()),
    ("url", pa.string()),
    ("metadata_json", pa.string()),
])


def to_string(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def make_row(record):
    metadata = record.get("metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    text = record["text"]

    return {
        "id": to_string(record.get("id")),
        "text": text,
        "source": to_string(record.get("source")),
        "created": to_string(record.get("created")),
        "added": to_string(record.get("added")),
        "language": record.get("language"),
        "text_length": len(text),
        "license": to_string(metadata.get("license")),
        "url": to_string(metadata.get("url")),
        "metadata_json": json.dumps(
            metadata,
            ensure_ascii=False,
            default=str,
        ),
    }


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_name = tempfile.mkstemp(
        suffix=".parquet",
        dir=OUTPUT_FILE.parent,
    )
    os.close(temp_fd)
    temp_file = Path(temp_name)

    writer = None
    batch = []

    stats = Counter()
    total = 0

    try:
        writer = pq.ParquetWriter(
            temp_file,
            SCHEMA,
            compression="snappy",
        )

        print("Starting dataset preprocessing...")

        with gzip.open(RAW_FILE, "rt", encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue

                total += 1

                processed = preprocess_record(record)

                if processed is None:
                    stats["filtered"] += 1
                    continue

                batch.append(make_row(processed))
                stats["accepted"] += 1

                if len(batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=SCHEMA)
                    writer.write_table(table)
                    batch.clear()

                if total % 500 == 0:
                    print(
                        f"Processed: {total} | "
                        f"Accepted: {stats['accepted']} | "
                        f"Filtered: {stats['filtered']}"
                    )

        if batch:
            table = pa.Table.from_pylist(batch, schema=SCHEMA)
            writer.write_table(table)

        writer.close()
        writer = None

        if stats["accepted"] == 0:
            raise RuntimeError("No records passed preprocessing.")

        os.replace(temp_file, OUTPUT_FILE)

        print("\n--- PREPROCESSING SUMMARY ---")
        print(f"Raw records:       {total}")
        print(f"Accepted records:  {stats['accepted']}")
        print(f"Filtered records:  {stats['filtered']}")
        print(f"Invalid JSON:      {stats['invalid_json']}")
        print(f"Output:            {OUTPUT_FILE}")
        print(
            f"Output size:       "
            f"{OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB"
        )
        print("\nPreprocessing completed.")

    except Exception:
        if writer is not None:
            writer.close()

        if temp_file.exists():
            temp_file.unlink()

        raise


if __name__ == "__main__":
    main()
