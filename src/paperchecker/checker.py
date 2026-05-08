"""LLM-powered citation verification.

Sends a claim + numbered source phrases to the LLM and parses
a 1-5 confidence score with supporting phrase index and reason.
"""

import re
from dataclasses import dataclass, field

from paperchecker.config import Config
from paperchecker.llm import call_llm
from paperchecker.phraser import split_phrases, format_numbered_phrases


def guess_doi(
    config: Config,
    title: str,
    author: str,
    year: str,
) -> str | None:
    """Ask the LLM to guess the DOI for a paper given its metadata.

    Returns the DOI string (e.g., '10.1234/example') or None.
    """
    prompt = f"""Find the DOI (Digital Object Identifier) for this academic paper.
Reply with ONLY the DOI string on one line (e.g., 10.1234/example.2025) or "UNKNOWN":

Title: {title}
Author: {author}
Year: {year}

DOI:"""

    resp = call_llm(config, prompt)
    if not resp:
        return None

    resp = resp.strip()
    if resp.upper() == "UNKNOWN" or not resp:
        return None

    # Extract DOI-like pattern from response
    import re

    m = re.search(r"(10\.\d{4,}/[^\s]+)", resp)
    if m:
        return m.group(1).rstrip(".")
    return None

VERIFICATION_PROMPT = """Does this source document substantiate the following claim made in a paper?

CLAIM: {claim}
{co_cited}

SOURCE: {source_name}
NUMBERED PHRASES:
{numbered_phrases}

If the source substantiates the claim, rate your confidence on this scale:
1 = no content suggesting the claim
2 = some related information but insufficient for citation
3 = related but not strongly supported
4 = supported with some question of interpretation
5 = exact match of the cited content

Reply in this format:
VERIFIED | confidence=N | phrase=[N] or phrase=[N-M] | reason=<one sentence>
or
UNSUBSTANTIATED | confidence=N | reason=<one sentence>

where N is your confidence level (1-5).

Answer:"""


@dataclass
class VerificationResult:
    """Result of verifying a claim against source text."""

    claim: str
    source_name: str
    evidence_found: bool
    confidence: int
    reason: str
    phrase_index: int = -1
    phrase_text: str | None = None
    co_cited: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "verified" if self.evidence_found else "unsubstantiated"


def verify_claim(
    config: Config,
    claim: str,
    source_text: str,
    source_name: str,
    backend: str | None = None,
    co_cited: list[str] | None = None,
) -> VerificationResult | None:
    """Verify a claim against source text using an LLM.

    Args:
        config: Application configuration.
        claim: The claim text to verify.
        source_text: The full source text to check against.
        source_name: Human-readable name of the source.
        backend: Specific LLM backend to use (uses preferred if None).
        co_cited: Other papers cited alongside this one on the same claim.

    Returns:
        VerificationResult with confidence score, or None if LLM call fails.
    """
    if not config.preferred_backend:
        return None

    phrases = split_phrases(source_text)
    numbered = format_numbered_phrases(phrases)

    co_cited_text = ""
    if co_cited:
        names = ", ".join(co_cited)
        co_cited_text = (
            f"NOTE: This claim cites multiple papers simultaneously. "
            f"You are checking whether THIS specific paper {source_name} "
            f"supports the claim. The claim may rely on other co-cited papers: [{names}]. "
            f"Score based on what THIS paper contributes."
        )

    prompt = VERIFICATION_PROMPT.format(
        claim=claim,
        co_cited=co_cited_text,
        source_name=source_name,
        numbered_phrases=numbered,
    )

    resp = call_llm(config, prompt, backend=backend)
    if not resp:
        return None

    return _parse_response(resp, claim, source_name, phrases, co_cited or [])


def _parse_response(
    response: str,
    claim: str,
    source_name: str,
    phrases: list[tuple[int, str]],
    co_cited: list[str] | None = None,
) -> VerificationResult:
    """Parse the LLM verification response."""
    parts = [p.strip() for p in response.split("|")]
    label = parts[0].strip().upper() if parts else ""
    evidence_found = label.startswith("VERIFIED")

    confidence = 0
    phrase_index = -1
    reason = ""

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("confidence="):
            try:
                confidence = int(part.split("=")[1].strip())
            except ValueError:
                pass
        elif part.startswith("phrase="):
            try:
                val = part.split("=")[1].strip()
                pm = re.match(r"\[(\d+)(?:-(\d+))?\]", val)
                if pm:
                    phrase_index = int(pm.group(1))
                else:
                    phrase_index = int(val)
            except (ValueError, IndexError):
                pass
        elif part.startswith("reason="):
            reason = part.split("=", 1)[1].strip()
        elif part and "reason" not in part and "phrase" not in part:
            if not reason:
                reason = part

    if not reason:
        reason = response

    phrase_text = None
    if 0 <= phrase_index < len(phrases):
        phrase_text = phrases[phrase_index][1]

    return VerificationResult(
        claim=claim,
        source_name=source_name,
        evidence_found=evidence_found,
        confidence=max(1, min(5, confidence or 3)),
        reason=reason,
        phrase_index=phrase_index,
        phrase_text=phrase_text,
        co_cited=co_cited or [],
    )
