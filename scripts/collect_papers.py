import gzip
import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset

OUTPUT_FILE = Path("data/raw/arxiv_papers.jsonl.gz")
CHECKPOINT = Path("data/raw/arxiv_papers_6000_checkpoint.jsonl")

TARGET_TOTAL = 6000
SAVE_INTERVAL = 100


def load_checkpoint():
    ids = set()

    if not CHECKPOINT.exists():
        return ids

    with CHECKPOINT.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                paper_id = record.get("id")

                if paper_id:
                    ids.add(paper_id)

            except json.JSONDecodeError:
                continue

    return ids


def main():
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    collected_ids = load_checkpoint()
    collected = len(collected_ids)

    print(f"Existing checkpoint records: {collected}")

    if collected >= TARGET_TOTAL:
        print("Target already collected. Preparing compressed output.")
    else:
        print("Connecting to Hugging Face dataset...")

        dataset = load_dataset(
            "common-pile/arxiv_papers",
            split="train",
            streaming=True,
            token=True,
        )

        print("Streaming started.")

        with CHECKPOINT.open("a", encoding="utf-8") as output:
            for record in dataset:
                paper_id = record.get("id")
                paper_text = record.get("text")

                if not paper_id or paper_id in collected_ids:
                    continue

                if not isinstance(paper_text, str) or not paper_text.strip():
                    continue

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    ) + "\n"
                )

                collected_ids.add(paper_id)
                collected += 1

                if collected % SAVE_INTERVAL == 0:
                    output.flush()
                    os.fsync(output.fileno())
                    print(f"Progress: {collected}/{TARGET_TOTAL}")

                if collected >= TARGET_TOTAL:
                    break

    if collected != TARGET_TOTAL:
        raise RuntimeError(
            f"Dataset ended before target: {collected}/{TARGET_TOTAL}"
        )

    print("Compressing collected papers...")

    with tempfile.NamedTemporaryFile(
        suffix=".gz",
        dir=OUTPUT_FILE.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        with CHECKPOINT.open("r", encoding="utf-8") as source:
            with gzip.open(
                temporary_path,
                "wt",
                encoding="utf-8",
            ) as destination:
                for line in source:
                    destination.write(line)

        os.replace(temporary_path, OUTPUT_FILE)
        CHECKPOINT.unlink()

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print("\nCollection completed successfully.")
    print(f"Unique papers: {collected}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
