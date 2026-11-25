import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

# Keep language detection results consistent between runs.
DetectorFactory.seed = 0

MIN_TEXT_LENGTH = 200
LANGUAGE_SAMPLE_LENGTH = 1000


def clean_text(text: str) -> str:
    """
    Remove HTML, normalize Unicode, and clean extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags and their contents where appropriate.
    soup = BeautifulSoup(text, "html.parser")

    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(separator=" ")

    # Replace repeated whitespace with a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    Returns 'unknown' when detection is not possible.
    """
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "unknown"

    try:
        return detect(text[:LANGUAGE_SAMPLE_LENGTH])
    except LangDetectException:
        return "unknown"


def preprocess_record(
    record: dict[str, Any],
    allowed_languages: set[str] | None = None,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> dict[str, Any] | None:
    """
    Clean one paper and filter invalid or unsupported records.

    Returns a cleaned record or None if the record should be excluded.
    """
    if allowed_languages is None:
        allowed_languages = {"en"}

    paper_id = record.get("id")
    raw_text = record.get("text")

    if not paper_id or not isinstance(raw_text, str):
        return None

    cleaned = clean_text(raw_text)

    if len(cleaned) < min_text_length:
        return None

    language = detect_language(cleaned)

    if language not in allowed_languages:
        return None

    # Preserve original fields and metadata.
    processed_record = record.copy()
    processed_record["text"] = cleaned
    processed_record["language"] = language

    return processed_record
