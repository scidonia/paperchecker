"""Manifest manager — creates and updates MD manifest files tracking verification state."""

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CitationEntry:
    """A single citation row in a manifest."""

    index: int
    claim: str
    citation_key: str
    source_paper: str = ""
    status: str = "unchecked"
    confidence: int = 0
    phrase: str = ""


@dataclass
class Manifest:
    """A manifest tracking one paper being verified."""

    paper_title: str
    source_path: str = ""
    extracted_text_path: str = ""
    status: str = "in_progress"
    citations: list[CitationEntry] = field(default_factory=list)
    manifest_dir: str = "manifests"

    @property
    def filename(self) -> str:
        """Generate a safe filename from the paper title."""
        safe = self.paper_title.lower().replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c == "_")
        return f"{safe}.md"

    @property
    def path(self) -> str:
        return os.path.join(self.manifest_dir, self.filename)


def create_manifest(
    paper_title: str,
    source_path: str,
    manifest_dir: str = "manifests",
) -> Manifest:
    """Create a new manifest for a paper."""
    return Manifest(
        paper_title=paper_title,
        source_path=source_path,
        manifest_dir=manifest_dir,
    )


def add_citation(
    manifest: Manifest,
    claim: str,
    citation_key: str,
    source_paper: str = "",
) -> CitationEntry:
    """Add a citation entry to the manifest."""
    idx = len(manifest.citations) + 1
    entry = CitationEntry(
        index=idx,
        claim=claim,
        citation_key=citation_key,
        source_paper=source_paper,
    )
    manifest.citations.append(entry)
    return entry


def update_citation(
    entry: CitationEntry,
    status: str | None = None,
    confidence: int | None = None,
    phrase: str | None = None,
) -> None:
    """Update a citation entry with verification results."""
    if status is not None:
        entry.status = status
    if confidence is not None:
        entry.confidence = confidence
    if phrase is not None:
        entry.phrase = phrase


def write_manifest(manifest: Manifest) -> str:
    """Write a manifest to disk as a Markdown file. Returns the path."""
    os.makedirs(manifest.manifest_dir, exist_ok=True)
    path = manifest.path

    lines = [
        f"# Paper: {manifest.paper_title}",
        "",
        f"**Source**: `{manifest.source_path}`",
        f"**Extracted Text**: `{manifest.extracted_text_path}`",
        f"**Status**: `{manifest.status}`",
        f"**Updated**: {datetime.now().isoformat()}",
        "",
        "## Citations",
        "",
        "| # | Claim | Citation Key | Source Paper | Status | Confidence | Phrase |",
        "|---|-------|-------------|-------------|--------|------------|--------|",
    ]

    for cit in manifest.citations:
        claim = cit.claim[:100].replace("|", "\\|")
        key = cit.citation_key[:50].replace("|", "\\|")
        source = cit.source_paper[:50].replace("|", "\\|")
        lines.append(
            f"| {cit.index} | {claim} | {key} | {source} "
            f"| {cit.status} | {cit.confidence} | {cit.phrase} |"
        )

    lines.append("")
    content = "\n".join(lines)

    with open(path, "w") as f:
        f.write(content)

    return path
