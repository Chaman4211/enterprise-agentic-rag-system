from typing import Any


DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40


def chunk_text(
    text: str,
    tokenizer: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping token-aware chunks.

    Character offsets preserve the original text instead of decoding
    token IDs, which helps retain formulas and punctuation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    if not isinstance(text, str) or not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks = []
    stride = chunk_size - chunk_overlap

    for start in range(0, len(offsets), stride):
        end = min(start + chunk_size, len(offsets))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = text[char_start:char_end]

        if chunk.strip():
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "text": chunk,
                    "token_count": end - start,
                    "start_token": start,
                    "end_token": end,
                }
            )

        if end == len(offsets):
            break

    return chunks
