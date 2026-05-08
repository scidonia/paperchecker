r"""Citation puller — reads LaTeX, extracts citations and claims.

Reads a .tex file, finds all \cite{} and \footnote{} commands,
and extracts the surrounding claim sentence for each citation.
"""

import re
import os
import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class Citation:
    """A single citation extracted from a LaTeX file."""

    citation_key: str
    claim: str
    tex_file: str
    line_number: int = 0

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(f"{self.claim}|{self.citation_key}".encode()).hexdigest()


def _find_brace_content(text: str, start: int) -> str:
    """Extract content from braces starting at position `start`.

    Handles nested braces — returns content between outer braces.
    `start` should point to the character after the opening brace.
    """
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth == 0:
        return text[start : end - 1]
    return text[start:]


def _find_sentence_start(text: str, pos: int) -> int:
    """Find the start of the sentence containing position `pos`."""
    for i in range(pos, 0, -1):
        if text[i] in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
            return i + 2
    return 0


def _clean_latex(s: str) -> str:
    """Strip LaTeX commands and whitespace from a string, preserving math mode."""
    # Handle escaped characters like \o, \O
    s = s.replace("{\\o}", "ø").replace("{\\O}", "Ø")
    s = s.replace("\\o ", "ø").replace("\\O ", "Ø")

    # Preserve math mode blocks ($...$) before stripping commands
    math_blocks: list[str] = []

    def _save_math(m: re.Match) -> str:
        math_blocks.append(m.group(0))
        return f"__MATH_{len(math_blocks) - 1}__"

    s = re.sub(r"\$[^$]+\$", _save_math, s)

    # Strip LaTeX commands from non-math text
    s = re.sub(r"\\[a-z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-z]+", "", s)
    s = re.sub(r"\{([^}]*)\}", r"\1", s)

    # Restore math blocks
    for i, block in enumerate(math_blocks):
        s = s.replace(f"__MATH_{i}__", block)

    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_citations(tex_path: str) -> list[Citation]:
    """Extract all citations from a LaTeX file, resolving \\input{} and \\include{}."""

    with open(tex_path) as f:
        text = f.read()

    text = _resolve_includes(text, os.path.dirname(tex_path))

    citations: list[Citation] = []

    for pattern in [r"\\cite\{", r"\\footnote\{"]:
        for m in re.finditer(pattern, text):
            content = _find_brace_content(text, m.end())
            sentence_start = _find_sentence_start(text, m.start())
            claim = _clean_latex(text[sentence_start : m.start()])
            # Split on commas for multi-citation commands like \cite{key1,key2}
            keys = [k.strip() for k in content.split(",") if k.strip()]
            for key in keys:
                clean_key = _clean_latex(key)[:300]
                citations.append(
                    Citation(
                        citation_key=clean_key,
                        claim=claim[:300],
                        tex_file=tex_path,
                        line_number=text[: m.start()].count("\n") + 1,
                    )
                )

    return citations


def _resolve_includes(text: str, base_dir: str, depth: int = 0) -> str:
    """Recursively resolve \\input{} and \\include{} commands in LaTeX text.

    Replaces each command with the content of the referenced file.
    Limits recursion depth to avoid cycles.
    """
    if depth > 10:
        return text

    # Find all \input{...} / \include{...} commands and their full ranges
    replacements: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\\(input|include)\{", text):
        brace_start = m.end() - 1  # position of '{'
        _find_brace_content(text, brace_start + 1)  # just advances internal state
        # Now find the matching closing brace
        brace_end = _find_closing_brace(text, brace_start)
        if brace_end < 0:
            continue
        content = text[brace_start + 1 : brace_end]
        filename = content.strip()
        if not filename.endswith(".tex"):
            filename = filename + ".tex"
        filepath = os.path.join(base_dir, filename)
        replacement = ""
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    sub_text = f.read()
                sub_dir = os.path.dirname(filepath)
                replacement = _resolve_includes(sub_text, sub_dir, depth + 1)
            except OSError:
                pass
        replacements.append((m.start(), brace_end + 1, replacement))

    # Apply replacements from right to left to preserve positions
    result = text
    for start, end, repl in reversed(replacements):
        result = result[:start] + repl + result[end:]

    return result


def _find_closing_brace(text: str, brace_start: int) -> int:
    """Find the position of the closing brace matching the brace at brace_start.

    Returns -1 if no matching brace found.
    """
    if brace_start >= len(text) or text[brace_start] != "{":
        return -1
    depth = 1
    end = brace_start + 1
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth == 0:
        return end - 1
    return -1


def load_cache(cache_dir: str, tex_path: str) -> dict:
    """Load the verification cache for a given tex file."""
    cache_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(tex_path))[0]}_cache.json"
    )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def resolve_source(
    citation_key: str, papers_dir: str, tex_path: str | None = None
) -> str | None:
    """Find a source file in papers_dir matching a citation key.

    Resolution order:
    1. If a refs.bib exists alongside the tex file, look up the bibtex key
       and match by author/year terms.
    2. Fall back to direct filename matching.
    """
    if not os.path.isdir(papers_dir):
        return None

    if tex_path:
        bib_path = _find_bib_path(tex_path)
        if bib_path:
            result = _resolve_via_bib(citation_key, bib_path, papers_dir)
            if result:
                return result

    return _resolve_via_filename(citation_key, papers_dir)


def get_citation_info(
    citation_key: str, tex_path: str
) -> dict[str, str]:
    """Extract bibtex metadata (author, title, year) for a citation key.

    Used for building search queries when the paper isn't found locally.
    """
    bib_path = _find_bib_path(tex_path)
    if not bib_path:
        return {}

    try:
        with open(bib_path) as f:
            bib_text = f.read()
    except OSError:
        return {}

    import re

    clean_key = citation_key.strip()
    pattern = r"@\w+\{" + re.escape(clean_key) + r","
    m = re.search(pattern, bib_text)
    if not m:
        return {}

    entry_text = _find_brace_content(bib_text, m.end() - 1)
    if not entry_text:
        return {}

    info: dict[str, str] = {}

    authors = _find_field_value(entry_text, "author")
    if authors:
        first = authors.split(" and ")[0].strip()
        parts = first.split(",")
        raw_author = parts[0].strip() if len(parts) > 1 else first.split()[-1].strip()
        # Strip LaTeX commands from author name (e.g., S{\o}rensen -> Sørensen)
        info["author"] = _clean_latex(raw_author)

    titles = _find_field_value(entry_text, "title")
    if titles:
        title = re.sub(r"\{+([^{}]+)\}+", r"\1", titles)
        title = re.sub(r"\s+", " ", title).strip()
        info["title"] = title[:200]

    years = re.findall(r"year\s*=\s*\{?(\d{4})\}?", entry_text)
    if years:
        info["year"] = years[0]

    return info


def _find_field_value(entry_text: str, field_name: str) -> str | None:
    """Extract a bibtex field value handling nested braces."""
    import re

    m = re.search(rf"{re.escape(field_name)}\s*=\s*\{{", entry_text)
    if not m:
        return None
    brace_pos = m.end() - 1
    content = _find_brace_content(entry_text, brace_pos + 1)
    return content.strip() if content else None


def build_search_query(citation_key: str, tex_path: str) -> str | None:
    """Build a search query from bibtex metadata for downloading."""
    info = get_citation_info(citation_key, tex_path)
    if not info:
        return None

    parts = []
    if info.get("author"):
        parts.append(info["author"])
    if info.get("title"):
        parts.append(info["title"])
    if info.get("year"):
        parts.append(info["year"])

    return " ".join(parts) if parts else None


def _find_bib_path(tex_path: str) -> str | None:
    """Find the bib file via \\bibliography{} command or by naming convention."""
    tex_dir = os.path.dirname(os.path.abspath(tex_path))

    # Scan the tex file (and its includes) for \bibliography{path}
    with open(tex_path) as f:
        text = f.read()

    for m in re.finditer(r"\\bibliography\{", text):
        content = _find_brace_content(text, m.end())
        bib_name = content.strip()
        if not bib_name.endswith(".bib"):
            bib_name = bib_name + ".bib"
        bib_path = os.path.join(tex_dir, bib_name)
        bib_path = os.path.normpath(bib_path)
        if os.path.exists(bib_path):
            return bib_path

    # Convention: refs.bib in same or parent directory
    bib_path = os.path.join(tex_dir, "refs.bib")
    if os.path.exists(bib_path):
        return bib_path
    parent_bib = os.path.join(os.path.dirname(tex_dir), "refs.bib")
    if os.path.exists(parent_bib):
        return parent_bib
    return None


def _resolve_via_bib(
    citation_key: str, bib_path: str, papers_dir: str
) -> str | None:
    """Resolve a citation key via its bibtex entry."""
    import re

    try:
        with open(bib_path) as f:
            bib_text = f.read()
    except OSError:
        return None

    clean_key = citation_key.strip()

    # Find @type{key, ...} using brace counting (handles nested braces)
    pattern = r"@\w+\{" + re.escape(clean_key) + r","
    m = re.search(pattern, bib_text)
    if not m:
        return None

    start = m.end()  # content after "@type{key,"
    entry_text = _find_brace_content(bib_text, m.end() - 1)
    if not entry_text:
        return None

    authors = re.findall(r"author\s*=\s*\{(.*?)\}", entry_text, re.DOTALL)
    years = re.findall(r"year\s*=\s*\{?(\d{4})\}?", entry_text)

    search_terms: list[str] = []
    if authors:
        names = authors[0].split(" and ")
        for name in names[:2]:
            parts = name.strip().split(",")
            if len(parts) > 1:
                search_terms.append(parts[0].strip().lower())
            else:
                words = name.strip().split()
                if words:
                    search_terms.append(words[-1].strip().lower())
    if years:
        search_terms.append(years[0])

    if not search_terms:
        return None

    return _match_file(search_terms, papers_dir)


def _resolve_via_filename(citation_key: str, papers_dir: str) -> str | None:
    """Resolve a citation key by direct filename matching.

    Splits the key into tokens on:
    - letter-to-digit boundaries (e.g., 'zhang2024' → 'zhang', '2024')
    - existing spaces, underscores, hyphens
    """
    import re

    key_lower = citation_key.lower()
    # Split on letter→digit and digit→letter transitions
    tokens = re.split(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", key_lower)
    # Also split on common separators within each token
    search_terms = []
    for token in tokens:
        # Skip very short tokens (less than 3 chars) unless they're digits
        for subtok in re.split(r"[\s_\-]+", token):
            subtok = subtok.strip()
            if subtok and (len(subtok) >= 3 or subtok.isdigit()):
                search_terms.append(subtok)
    return _match_file(search_terms, papers_dir)


def _match_file(search_terms: list[str], papers_dir: str) -> str | None:
    """Score files in papers_dir against search terms.

    For each file, count how many search terms appear in its name.
    Return the best match if the score meets a minimum threshold.
    """
    best_score = 0
    best_path = None

    for f in os.listdir(papers_dir):
        if not f.endswith((".pdf", ".txt")) or f == "BIBLIOGRAPHY.md":
            continue
        f_lower = f.lower()
        score = 0
        for term in search_terms:
            if term and term in f_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_path = os.path.join(papers_dir, f)

    if best_score >= max(1, len(search_terms) * 0.5):
        return best_path
    return None


def save_cache(cache_dir: str, tex_path: str, cache: dict) -> None:
    """Save the verification cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(tex_path))[0]}_cache.json"
    )
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
