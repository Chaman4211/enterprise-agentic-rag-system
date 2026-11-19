from src.preprocessing import (
    clean_text,
    detect_language,
    preprocess_record,
)


def test_clean_text_removes_html():
    text = "<p>Hello <b>world</b></p>"
    assert clean_text(text) == "Hello world"


def test_clean_text_removes_extra_whitespace():
    text = "Hello    world\n\nPython"
    assert clean_text(text) == "Hello world Python"


def test_clean_text_normalizes_unicode():
    text = "ＡＩ research"
    assert clean_text(text) == "AI research"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""


def test_detect_language_identifies_english():
    text = (
        "Artificial intelligence is a field of computer science. "
        "Machine learning allows computers to learn patterns from data. "
        "Researchers develop algorithms for many scientific applications."
    )
    assert detect_language(text) == "en"


def test_preprocess_record_preserves_metadata():
    record = {
        "id": "paper-001",
        "text": (
            "Artificial intelligence is a field of computer science. "
            "Machine learning allows computers to learn patterns from data. "
        ) * 10,
        "metadata": {"license": "Public Domain"},
        "source": "arxiv-papers",
    }

    result = preprocess_record(record)

    assert result is not None
    assert result["id"] == "paper-001"
    assert result["metadata"] == {"license": "Public Domain"}
    assert result["source"] == "arxiv-papers"
    assert result["language"] == "en"


def test_preprocess_record_rejects_missing_id():
    record = {
        "text": "Artificial intelligence research " * 30,
        "metadata": {},
    }

    assert preprocess_record(record) is None


def test_preprocess_record_rejects_short_text():
    record = {
        "id": "paper-002",
        "text": "Short text",
        "metadata": {},
    }

    assert preprocess_record(record) is None


def test_preprocess_record_rejects_non_english():
    record = {
        "id": "paper-003",
        "text": (
            "La inteligencia artificial es una rama de la informática. "
            "Los investigadores desarrollan algoritmos para aprender "
            "patrones y resolver problemas complejos."
        ) * 10,
        "metadata": {},
    }

    assert preprocess_record(record) is None
