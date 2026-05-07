"""Hash-based registry — tracks downloaded papers and verified citations.

Uses content hashing to:
- Avoid re-downloading papers we already have (by SHA256 of PDF content).
- Avoid re-checking citations we've already verified (by SHA256 of claim + source).
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def sha256_hex(content: str | bytes) -> str:
    """Compute SHA256 hex digest of content."""
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Paper Registry (avoids re-download)
# ---------------------------------------------------------------------------

PAPER_REGISTRY_FILENAME = "_paper_registry.json"


@dataclass
class PaperEntry:
    """Metadata about a downloaded paper."""

    content_hash: str
    filename: str
    source_url: str
    source_type: str  # "arxiv", "doi", "scholar"
    downloaded_at: str
    file_size: int = 0


def _load_paper_registry(registry_dir: str) -> dict[str, dict]:
    path = os.path.join(registry_dir, PAPER_REGISTRY_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_paper_registry(registry_dir: str, registry: dict) -> None:
    os.makedirs(registry_dir, exist_ok=True)
    path = os.path.join(registry_dir, PAPER_REGISTRY_FILENAME)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def lookup_paper(content: bytes, registry_dir: str) -> Optional[PaperEntry]:
    """Check if we already have a paper with this content hash."""
    content_hash = sha256_hex(content)
    registry = _load_paper_registry(registry_dir)
    entry = registry.get(content_hash)
    if entry:
        return PaperEntry(
            content_hash=content_hash,
            filename=entry["filename"],
            source_url=entry["source_url"],
            source_type=entry["source_type"],
            downloaded_at=entry["downloaded_at"],
            file_size=entry.get("file_size", 0),
        )
    return None


def register_paper(
    content: bytes,
    source_url: str,
    source_type: str,
    registry_dir: str,
    filename: str = "",
    file_size: int = 0,
) -> PaperEntry:
    """Register a downloaded paper in the registry."""
    content_hash = sha256_hex(content)
    registry = _load_paper_registry(registry_dir)
    entry = {
        "filename": filename,
        "source_url": source_url,
        "source_type": source_type,
        "downloaded_at": datetime.now().isoformat(),
        "file_size": file_size,
    }
    registry[content_hash] = entry
    _save_paper_registry(registry_dir, registry)
    return PaperEntry(
        content_hash=content_hash,
        filename=filename,
        source_url=source_url,
        source_type=source_type,
        downloaded_at=entry["downloaded_at"],
        file_size=file_size,
    )


# ---------------------------------------------------------------------------
# Check Cache (avoids re-checking citations)
# ---------------------------------------------------------------------------

CHECK_CACHE_FILENAME = "_check_cache.json"


def _load_check_cache(cache_dir: str) -> dict:
    path = os.path.join(cache_dir, CHECK_CACHE_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_check_cache(cache_dir: str, cache: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CHECK_CACHE_FILENAME)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def check_citation_hash(claim: str, citation_key: str, source_text: str) -> str:
    """Compute a cache key for a (claim, citation, source) triple."""
    combined = f"{claim}|{citation_key}|{sha256_hex(source_text)}"
    return sha256_hex(combined)


def lookup_check(cache_key: str, cache_dir: str) -> Optional[dict]:
    """Look up a previously cached verification result."""
    cache = _load_check_cache(cache_dir)
    return cache.get(cache_key)


def store_check(
    cache_key: str,
    status: str,
    confidence: int,
    reason: str,
    phrase: str,
    cache_dir: str,
) -> None:
    """Store a verification result in the check cache."""
    cache = _load_check_cache(cache_dir)
    cache[cache_key] = {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "phrase": phrase,
    }
    _save_check_cache(cache_dir, cache)
