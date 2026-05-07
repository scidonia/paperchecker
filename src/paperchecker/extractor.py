"""Text extractor — converts PDFs and other files to plaintext.

Uses a fallback chain: pypdf → pdfplumber → pdftotext → .txt files.
"""

import os
import re
import subprocess


def extract_text(path: str, max_chars: int | None = None) -> str | None:
    """Extract text from a PDF or text file.

    Args:
        path: Path to the file (PDF or .txt).
        max_chars: Maximum characters to return (optional). If None, returns full text.

    Returns:
        Extracted text as a string, or None if extraction fails.
    """
    if path.endswith(".txt"):
        with open(path) as f:
            content = f.read()

        # If content looks like HTML (Internet Archive often serves .txt as HTML)
        if content[:500].strip().startswith("<!DOCTYPE") or content[
            :500
        ].strip().startswith("<html"):
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

        if max_chars:
            return content[:max_chars]
        return content

    # Try pypdf
    result = _try_pypdf(path)
    if result:
        return _trim(result, max_chars)

    # Try pdfplumber
    result = _try_pdfplumber(path)
    if result:
        return _trim(result, max_chars)

    # Try pdftotext (system command)
    result = _try_pdftotext(path)
    if result:
        return _trim(result, max_chars)

    # Fallback: try .txt version of the same file
    txt_path = path.replace(".pdf", ".txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return _trim(f.read(), max_chars or 8000)

    return None


def _try_pypdf(path: str) -> str | None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass
    return None


def _try_pdfplumber(path: str) -> str | None:
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = " ".join(p.extract_text() or "" for p in pdf.pages)
            if text.strip():
                return text
    except Exception:
        pass
    return None


def _try_pdftotext(path: str) -> str | None:
    try:
        r = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except Exception:
        pass
    return None


def _trim(text: str, max_chars: int | None) -> str:
    if max_chars:
        return text[:max_chars]
    return text
