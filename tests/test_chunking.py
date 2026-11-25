import re

import pytest

from src.chunking import chunk_text


class SimpleTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))
        return {
            "offset_mapping": [
                (match.start(), match.end()) for match in matches
            ]
        }


@pytest.fixture
def tokenizer():
    return SimpleTokenizer()


def test_short_text_produces_one_chunk(tokenizer):
    chunks = chunk_text(
        "one two three",
        tokenizer,
        chunk_size=5,
        chunk_overlap=1,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "one two three"
    assert chunks[0]["token_count"] == 3


def test_long_text_respects_chunk_size(tokenizer):
    text = "one two three four five six seven eight nine ten"

    chunks = chunk_text(
        text,
        tokenizer,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk["token_count"] for chunk in chunks] == [4, 4, 4]


def test_adjacent_chunks_preserve_overlap(tokenizer):
    text = "one two three four five six seven"

    chunks = chunk_text(
        text,
        tokenizer,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks[0]["text"].split()[-1] == "four"
    assert chunks[1]["text"].split()[0] == "four"


def test_chunk_indexes_and_token_ranges_are_consistent(tokenizer):
    chunks = chunk_text(
        "one two three four five six seven",
        tokenizer,
        chunk_size=3,
        chunk_overlap=1,
    )

    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert chunks[0]["start_token"] == 0
    assert chunks[0]["end_token"] == 3
    assert chunks[1]["start_token"] == 2
    assert chunks[1]["end_token"] == 5


def test_empty_text_returns_no_chunks(tokenizer):
    assert chunk_text("", tokenizer) == []
    assert chunk_text("   ", tokenizer) == []


def test_chunking_preserves_latex_and_original_text(tokenizer):
    text = r"Equation: $E = mc^2$ and formula: \(a^2 + b^2 = c^2\)."

    chunks = chunk_text(
        text,
        tokenizer,
        chunk_size=20,
        chunk_overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == text


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (4, 4),
        (4, 5),
        (4, -1),
    ],
)
def test_invalid_chunk_configuration_raises_error(
    tokenizer,
    chunk_size,
    chunk_overlap,
):
    with pytest.raises(ValueError):
        chunk_text(
            "one two three four five",
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
