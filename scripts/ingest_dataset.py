import argparse
import gzip
import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "common-pile/arxiv_papers"


def load_existing_ids(output_path):
    existing_ids = set()

    if output_path.exists():
        print("Checking existing dataset for saved records...")

        with gzip.open(output_path, "rt", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    existing_ids.add(str(record["id"]))

    return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="Stream and save unique arXiv papers."
    )
    parser.add_argument("--target", type=int, default=5000)
    args = parser.parse_args()

    output_path = Path("data/raw/arxiv_papers.jsonl.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(output_path)
    saved_count = len(seen_ids)

    print(f"Existing unique records: {saved_count}")
    print(f"Target records: {args.target}")

    if saved_count >= args.target:
        print("Target already reached. Nothing to download.")
        return

    print("\nConnecting to Hugging Face dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    new_count = 0

    with gzip.open(output_path, "at", encoding="utf-8") as file:
        for record in dataset:
            record_id = record.get("id")
            paper_text = record.get("text")

            if not record_id or not isinstance(paper_text, str):
                continue

            record_id = str(record_id)

            if record_id in seen_ids or not paper_text.strip():
                continue

            json_record = {
                "id": record_id,
                "text": paper_text,
                "source": record.get("source"),
                "created": record.get("created"),
                "added": record.get("added"),
                "metadata": record.get("metadata") or {},
            }

            file.write(
                json.dumps(
                    json_record,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )

            seen_ids.add(record_id)
            new_count += 1
            saved_count += 1

            if new_count % 100 == 0:
                file.flush()

            if new_count % 250 == 0:
                print(
                    f"Progress: {saved_count}/{args.target} "
                    f"unique papers saved"
                )

            if saved_count >= args.target:
                break

    print("\nIngestion completed.")
    print(f"New records saved: {new_count}")
    print(f"Total unique records: {saved_count}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
