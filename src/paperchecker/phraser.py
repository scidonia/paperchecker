"""Phrase splitter — splits source text into numbered phrases using spacy."""


def split_phrases(
    text: str,
    min_length: int = 20,
    max_phrases: int | None = 500,
) -> list[tuple[int, str]]:
    """Split text into numbered phrases using spacy sentence segmentation.

    Args:
        text: The full source text.
        min_length: Minimum character length for a phrase to be included.
        max_phrases: Maximum number of phrases to return (default 500 for LLM context).

    Returns:
        List of (index, phrase) tuples, where index is 0-based.
    """
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model isn't installed, use basic splitting
        return _fallback_split(text, min_length, max_phrases)

    doc = nlp(text)
    phrases: list[tuple[int, str]] = []
    for i, sent in enumerate(doc.sents):
        phrase = sent.text.strip()
        if len(phrase) >= min_length:
            phrases.append((i, phrase))
        if max_phrases and len(phrases) >= max_phrases:
            break
    return phrases


def _fallback_split(
    text: str,
    min_length: int = 20,
    max_phrases: int | None = 500,
) -> list[tuple[int, str]]:
    """Fallback regex-based sentence splitting when spacy is unavailable."""
    import re

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= min_length
    ]
    if max_phrases:
        sentences = sentences[:max_phrases]
    return list(enumerate(sentences))


def format_numbered_phrases(
    phrases: list[tuple[int, str]],
    truncate: int | None = 300,
) -> str:
    """Format phrases into a numbered string for the LLM prompt.

    Args:
        phrases: List of (index, phrase) tuples.
        truncate: Maximum characters per phrase in the formatted output.

    Returns:
        Newline-separated numbered phrase string like:
        [0] First phrase text...
        [1] Second phrase text...
    """
    lines = []
    for idx, phrase in phrases:
        text = phrase[:truncate] if truncate else phrase
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)
