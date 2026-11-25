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



def test_clean_text_preserves_latex_formulas():
    text = (
        r"Einstein equation: $E = mc^2$. "
        r"Inline formula: \(a^2 + b^2 = c^2\). "
        r"Display formula: \[x^2 + y^2 = z^2\]."
    )

    cleaned = clean_text(text)

    assert "$E = mc^2$" in cleaned
    assert r"\(a^2 + b^2 = c^2\)" in cleaned
    assert r"\[x^2 + y^2 = z^2\]" in cleaned


def test_clean_text_preserves_latex_inside_html():
    text = r"<p>Equation: <span>$\alpha + \beta = \gamma$</span></p>"

    cleaned = clean_text(text)

    assert r"$\alpha + \beta = \gamma$" in cleaned


def test_detect_language_uses_first_1000_characters(monkeypatch):
    import src.preprocessing as preprocessing

    captured = {}

    def fake_detect(text):
        captured["text"] = text
        return "en"

    monkeypatch.setattr(preprocessing, "detect", fake_detect)

    text = "A" * 1500

    assert detect_language(text) == "en"
    assert len(captured["text"]) == 1000
    assert captured["text"] == text[:1000]
