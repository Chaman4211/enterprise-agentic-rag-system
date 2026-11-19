import json
from pathlib import Path

import pyarrow.parquet as pq

FILE = Path("data/processed/clean_corpus.parquet")

EXPECTED_COLUMNS = {
    "id",
    "text",
    "source",
    "created",
    "added",
    "language",
    "text_length",
    "license",
    "url",
    "metadata_json",
}


def main():
    parquet = pq.ParquetFile(FILE)
    schema = parquet.schema_arrow

    print("--- COMPLETE PARQUET SCHEMA ---")
    print(schema)

    actual_columns = set(schema.names)
    print("\n--- VALIDATION ---")

    missing_columns = EXPECTED_COLUMNS - actual_columns
    print(f"Missing columns: {missing_columns or 'None'}")

    total = 0
    unique_ids = set()
    invalid_metadata = 0
    missing_metadata = 0
    missing_text = 0
    wrong_language = 0

    for batch in parquet.iter_batches(
        columns=["id", "text", "language", "metadata_json"],
        batch_size=250,
    ):
        data = batch.to_pydict()

        for paper_id, text, language, metadata in zip(
            data["id"],
            data["text"],
            data["language"],
            data["metadata_json"],
        ):
            total += 1

            if paper_id:
                unique_ids.add(paper_id)

            if not text:
                missing_text += 1

            if language != "en":
                wrong_language += 1

            if not metadata:
                missing_metadata += 1
            else:
                try:
                    json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    invalid_metadata += 1

    print(f"Total records:       {total}")
    print(f"Unique IDs:          {len(unique_ids)}")
    print(f"Missing text:        {missing_text}")
    print(f"Non-English records: {wrong_language}")
    print(f"Missing metadata:    {missing_metadata}")
    print(f"Invalid metadata:    {invalid_metadata}")

    passed = (
        not missing_columns
        and total == 5965
        and len(unique_ids) == total
        and missing_text == 0
        and wrong_language == 0
        and invalid_metadata == 0
    )

    print(f"\nRESULT: {'PASS' if passed else 'CHECK REQUIRED'}")


if __name__ == "__main__":
    main()
