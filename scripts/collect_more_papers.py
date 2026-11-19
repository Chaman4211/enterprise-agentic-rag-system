import gzip
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

OUTPUT_FILE = Path("data/raw/arxiv_papers.jsonl.gz")
REPO_ID = "common-pile/arxiv_papers"
SHARD = "00021_arxiv-papers.jsonl.gz"

TARGET_TOTAL = 6000
PROGRESS_INTERVAL = 100


def load_existing_ids():
    ids = set()

    with gzip.open(OUTPUT_FILE, "rt", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("id"):
                ids.add(record["id"])

    return ids


def download_shard():
    for attempt in range(1, 6):
        try:
            print(f"Downloading shard: {SHARD}")
            return hf_hub_download(
                repo_id=REPO_ID,
                filename=SHARD,
                repo_type="dataset",
                token=True,
            )
        except Exception as error:
            print(f"Download attempt {attempt}/5 failed: {error}")

            if attempt == 5:
                raise

            time.sleep(attempt * 10)


def main():
    existing_ids = load_existing_ids()

    print(f"Existing unique papers: {len(existing_ids)}")

    if len(existing_ids) >= TARGET_TOTAL:
        print("Target already reached.")
        return

    shard_path = download_shard()

    print("Shard downloaded. Reading locally...")
    added = 0

    with gzip.open(shard_path, "rt", encoding="utf-8") as source:
        with gzip.open(OUTPUT_FILE, "at", encoding="utf-8") as output:

            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                paper_id = record.get("id")
                text = record.get("text", "")

                if not paper_id or paper_id in existing_ids:
                    continue

                if not isinstance(text, str) or not text.strip():
                    continue

                output.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )

                existing_ids.add(paper_id)
                added += 1

                if added % PROGRESS_INTERVAL == 0:
                    output.flush()
                    print(
                        f"Progress: {len(existing_ids)}/{TARGET_TOTAL} "
                        f"unique papers"
                    )

                if len(existing_ids) >= TARGET_TOTAL:
                    break

    print("\nCollection completed.")
    print(f"New papers collected: {added}")
    print(f"Total unique papers: {len(existing_ids)}")
    print(f"Output size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
