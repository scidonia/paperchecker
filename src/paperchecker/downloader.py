"""Paper downloader — uses playwright to fetch papers from known repositories.

Supports downloading from:
- arXiv (by ID, title search)
- DOI resolvers (doi.org)
- Google Scholar (search result link extraction)

Uses a content-hash registry to avoid re-downloading papers we already have.
"""

import os
import re
from typing import Optional

from paperchecker.registry import lookup_paper, register_paper

REGISTRY_DIR = "papers"


def download_from_arxiv(
    arxiv_id: str,
    output_dir: str = "papers",
    headless: bool = True,
) -> str | None:
    """Download a paper PDF from arXiv by ID.

    Args:
        arxiv_id: arXiv paper ID (e.g., '2401.12345' or '2401.12345v1').
        output_dir: Directory to save the PDF.
        headless: Whether to run browser headless.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    filename = f"{arxiv_id}.pdf"

    # Try direct HTTP first (faster, no browser)
    result = _download_via_http(pdf_url, output_dir, filename, "arxiv")
    if result:
        return result

    # Fall back to browser-based download
    return _download_pdf(pdf_url, output_dir, filename, "arxiv", headless)


def search_arxiv(
    query: str,
    output_dir: str = "papers",
    expected_title: str | None = None,
    expected_year: str | None = None,
    headless: bool = True,
) -> str | None:
    """Search arXiv by query and download the best matching PDF.

    When expected_title/expected_year are provided, result titles are scored
    by word overlap and year is checked to avoid downloading wrong papers.
    """
    import json
    from urllib.parse import quote
    from urllib.request import urlopen, Request

    encoded = quote(query, safe="")
    api_url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&max_results=5"
    )
    try:
        req = Request(api_url, headers={"User-Agent": "paperchecker/0.1.0"})
        resp = urlopen(req, timeout=30)
        data = resp.read().decode()
    except Exception:
        return download_from_scholar(query, output_dir, headless)

    import re

    # Extract IDs and titles from the Atom response
    entries = re.findall(r"<entry>(.*?)</entry>", data, re.DOTALL)
    candidates: list[tuple[float, str, str]] = []  # (score, arxiv_id, title)

    for entry in entries:
        ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", entry)
        titles = re.findall(r"<title>([^<]+)</title>", entry)
        if not ids or not titles:
            continue
        arxiv_id = ids[0].strip()
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]
        result_title = titles[0].strip().replace("\n", " ")

        score = 1.0
        if expected_title:
            score = _title_similarity(expected_title, result_title)

        # If we have an expected year, check the arXiv ID year (YYMM)
        if expected_year:
            arxiv_year = "20" + arxiv_id[:2]
            if arxiv_year != expected_year:
                score *= 0.5  # penalise year mismatch

        candidates.append((score, arxiv_id, result_title))

    # Sort by score descending
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Try candidates with score above threshold
    for score, arxiv_id, result_title in candidates:
        if score < 0.4:
            continue
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        filename = f"{arxiv_id}.pdf"
        path = _download_via_http(pdf_url, output_dir, filename, "arxiv")
        if path:
            return path

    return None


def search_semantic_scholar(
    query: str,
    output_dir: str = "papers",
    expected_title: str | None = None,
    expected_year: str | None = None,
) -> str | None:
    """Search Semantic Scholar by query and download the best matching PDF.

    Uses the free Semantic Scholar API to find papers across all years
    and disciplines, including pre-1990 works not on arXiv.

    Args:
        query: Search query (author, title, etc.).
        output_dir: Directory to save the PDF.
        expected_title: Optional expected paper title for result filtering.
        expected_year: Optional expected publication year for filtering.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    import json
    from urllib.parse import quote
    from urllib.request import urlopen, Request

    fields = "title,year,externalIds,openAccessPdf,isOpenAccess"
    encoded = quote(query, safe="")
    api_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded}&year={expected_year or ''}&limit=5&fields={fields}"
    )
    try:
        import time

        # Retry with exponential backoff for rate limits
        for attempt in range(3):
            time.sleep(1.0 * (attempt + 1))
            req = Request(api_url, headers={"User-Agent": "paperchecker/0.1.0"})
            try:
                resp = urlopen(req, timeout=30)
                data = json.loads(resp.read().decode())
                break
            except Exception:
                if attempt == 2:
                    return None
                continue
    except Exception:
        return None

    papers = data.get("data", [])
    candidates: list[tuple[float, dict]] = []

    for paper in papers:
        result_title = paper.get("title") or ""
        result_year = str(paper.get("year") or "")

        score = 1.0
        if expected_title and result_title:
            score = _title_similarity(expected_title, result_title)
        if expected_year and result_year and result_year != expected_year:
            score *= 0.5

        if score >= 0.4:
            candidates.append((score, paper))

    candidates.sort(key=lambda c: c[0], reverse=True)

    for score, paper in candidates:
        # Try open access PDF first
        oa = paper.get("openAccessPdf")
        if oa and oa.get("url"):
            ext_ids = paper.get("externalIds") or {}
            paper_id = ext_ids.get("DOI") or ext_ids.get("ArXiv") or "ss"
            filename = _safe_filename(paper_id) + ".pdf"
            path = _download_via_http(oa["url"], output_dir, filename, "semantic_scholar")
            if path:
                return path

        # Try DOI via sci-hub
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        if doi:
            path = _download_via_scihub(doi, output_dir)
            if path:
                return path

        # Try DOI via Unpaywall for open access alternatives
        if doi:
            path = _download_via_unpaywall(doi, output_dir)
            if path:
                return path

        # Try arXiv ID directly
        arxiv_id = ext_ids.get("ArXiv")
        if arxiv_id:
            path = download_from_arxiv(arxiv_id, output_dir, headless=False)
            if path:
                return path

    return None


def search_crossref(
    query: str,
    output_dir: str = "papers",
    expected_title: str | None = None,
    expected_year: str | None = None,
) -> str | None:
    """Search CrossRef API for a paper by title/author, get DOI, download via sci-hub.

    CrossRef indexes major publishers including ScienceDirect/Elsevier,
    Springer, ACM, and IEEE. Free API with good rate limits.

    Returns path to downloaded PDF or None.
    """
    import json
    from urllib.parse import quote
    from urllib.request import urlopen, Request

    encoded = quote(query, safe="")
    api_url = (
        "https://api.crossref.org/works?"
        f"query={encoded}&rows=5"
    )
    try:
        req = Request(
            api_url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            },
        )
        resp = urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
    except Exception:
        return None

    items = data.get("message", {}).get("items", [])
    candidates: list[tuple[float, str]] = []  # (score, doi)

    for item in items:
        doi = item.get("DOI")
        if not doi:
            continue

        # Extract title for scoring
        titles = item.get("title", [])
        result_title = titles[0] if titles else ""

        score = 1.0
        if expected_title and result_title:
            score = _title_similarity(expected_title, result_title)

        # Check year (only penalise for large mismatches)
        if expected_year:
            date_parts = (
                item.get("published-print", {})
                .get("date-parts", [[None]])
            )
            item_year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
            if item_year and item_year != expected_year:
                try:
                    gap = abs(int(item_year) - int(expected_year))
                    if gap <= 2:
                        score *= 1.0  # minor discrepancy, no penalty
                    elif gap <= 5:
                        score *= 0.8
                    else:
                        score *= 0.5
                except ValueError:
                    score *= 0.5

        if score >= 0.4:
            candidates.append((score, doi))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Try downloading each candidate DOI via sci-hub
    for _, doi in candidates:
        path = _download_via_scihub(doi, output_dir)
        if path:
            return path

    return None


def _safe_filename(s: str) -> str:
    """Convert a string (e.g., DOI) into a safe filename."""
    import re

    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    return s[:120]


def _title_similarity(expected: str, candidate: str) -> float:
    """Score how well a candidate title matches the expected title.

    Uses case-insensitive word overlap (Jaccard-like score).
    Handles LaTeX macros, newlines, and extra whitespace in both titles.
    """
    import re

    def _tokenize(s: str) -> set[str]:
        s = re.sub(r"\{[^}]*\}", "", s)  # remove braced LaTeX
        s = re.sub(r"\\[a-zA-Z]+", "", s)  # remove LaTeX commands
        s = re.sub(r"[^a-zA-Z0-9]+", " ", s)  # normalise non-word chars
        words = {w.lower() for w in s.split() if len(w) > 1}
        return words

    expected_words = _tokenize(expected)
    candidate_words = _tokenize(candidate)

    if not expected_words:
        return 1.0

    overlap = expected_words & candidate_words
    return len(overlap) / len(expected_words)


def _download_via_scihub(
    doi: str,
    output_dir: str,
) -> str | None:
    """Try to download a PDF via sci-hub using the DOI."""
    import re
    from urllib.request import urlopen, Request

    # Known sci-hub domains
    for domain in ["sci-hub.se", "sci-hub.ru", "sci-hub.st"]:
        sci_url = f"https://{domain}/{doi}"
        try:
            req = Request(
                sci_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "Referer": f"https://{domain}/",
                },
            )
            resp = urlopen(req, timeout=15)
            html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue

        # sci-hub embeds the PDF URL in a meta tag (citation_pdf_url)
        pdf_url = None
        m = re.search(
            r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']',
            html,
        )
        if m:
            pdf_url = m.group(1)
        else:
            # Try other patterns as fallback
            for pattern in [
                r'<iframe[^>]+src\s*=\s*["\']([^"\']+\.pdf[^"\']*)',
                r'<embed[^>]+src\s*=\s*["\']([^"\']+\.pdf[^"\']*)',
            ]:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    pdf_url = m.group(1)
                    break

        if not pdf_url:
            continue

        # Resolve relative URLs
        if pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url
        elif pdf_url.startswith("/"):
            pdf_url = f"https://{domain}" + pdf_url

        filename = _safe_filename(doi) + ".pdf"
        path = _download_via_http(pdf_url, output_dir, filename, "scihub")
        if path:
            return path

    return None


def _download_via_unpaywall(
    doi: str,
    output_dir: str,
) -> str | None:
    """Try to find and download an open-access PDF via Unpaywall API.

    Uses the free Unpaywall API to locate legitimate OA copies of a paper
    by DOI, then downloads the best available PDF.
    """
    import json
    from urllib.parse import quote
    from urllib.request import urlopen, Request

    encoded = quote(doi, safe="")
    api_url = f"https://api.unpaywall.org/v2/{encoded}?email=paperchecker@scidonia.ai"
    try:
        req = Request(api_url, headers={"User-Agent": "paperchecker/0.1.0"})
        resp = urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
    except Exception:
        return None

    # Try best OA location
    best = data.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url")
    if pdf_url:
        filename = _safe_filename(doi) + ".pdf"
        return _download_via_http(pdf_url, output_dir, filename, "unpaywall")

    # Try other OA locations
    for loc in data.get("oa_locations", []):
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if pdf_url:
            filename = _safe_filename(doi) + ".pdf"
            path = _download_via_http(pdf_url, output_dir, filename, "unpaywall")
            if path:
                return path

    return None
    """Convert a string (e.g., DOI) into a safe filename."""
    import re

    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    return s[:120]


def download_from_doi(
    doi: str,
    output_dir: str = "papers",
    headless: bool = True,
) -> str | None:
    """Download a paper PDF by resolving a DOI.

    Tries sci-hub first, then doi.org redirect, then Unpaywall.
    """
    # Try sci-hub first
    path = _download_via_scihub(doi, output_dir)
    if path:
        return path

    # Fall back to doi.org resolution + playwright
    doi_clean = doi.strip()
    if doi_clean.startswith("http"):
        doi_clean = doi_clean.split("doi.org/")[-1]

    url = f"https://doi.org/{doi_clean}"
    safe_name = _safe_filename(doi_clean)
    filename = f"{safe_name}.pdf"
    return _download_pdf(url, output_dir, filename, "doi", headless)


def download_from_scholar(
    query: str,
    output_dir: str = "papers",
    headless: bool = True,
) -> str | None:
    """Search Google Scholar and download the first result PDF.

    Args:
        query: Search query (author, title, etc.).
        output_dir: Directory to save the PDF.
        headless: Whether to run browser headless.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    from urllib.parse import quote_plus

    search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}"
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", query)[:80]
    filename = f"scholar_{safe_name}.pdf"
    return _download_pdf(search_url, output_dir, filename, "scholar", headless)


def _download_via_http(
    url: str,
    output_dir: str,
    filename: str,
    source_type: str,
) -> str | None:
    """Download a PDF directly via HTTP (no browser).

    Used for sources that serve PDFs directly, like arXiv.
    Checks the hash registry before writing.
    """
    from urllib.request import urlopen, Request
    from paperchecker.registry import lookup_paper, register_paper

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Referer": url.rsplit("/", 1)[0] + "/",
                "Accept": "application/pdf,*/*",
            },
        )
        resp = urlopen(req, timeout=60)
        content = resp.read()
    except Exception:
        return None

    if not content or len(content) < 1000:
        return None

    # Check if it's actually a PDF
    if not content[:5].startswith(b"%PDF"):
        return None

    # Check hash registry for duplicate
    existing = lookup_paper(content, output_dir)
    if existing and os.path.exists(os.path.join(output_dir, existing.filename)):
        return os.path.join(output_dir, existing.filename)

    with open(output_path, "wb") as f:
        f.write(content)

    register_paper(
        content,
        source_url=url,
        source_type=source_type,
        registry_dir=output_dir,
        filename=filename,
        file_size=len(content),
    )
    return output_path


def _download_pdf(
    url: str,
    output_dir: str,
    filename: str,
    source_type: str,
    headless: bool = True,
) -> str | None:
    """Download a PDF using playwright.

    Navigates to the URL, looks for PDF links or direct downloads,
    and saves the file. Checks content-hash registry first to avoid
    re-downloading papers we already have.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    def _save_and_register(content: bytes) -> str:
        """Save downloaded content and register it in the hash registry."""
        # Check if we already have a paper with this exact content (by hash)
        existing = lookup_paper(content, output_dir)
        if existing and os.path.exists(os.path.join(output_dir, existing.filename)):
            return os.path.join(output_dir, existing.filename)

        with open(output_path, "wb") as f:
            f.write(content)
        register_paper(
            content,
            source_url=url,
            source_type=source_type,
            registry_dir=output_dir,
            filename=filename,
            file_size=len(content),
        )
        return output_path

    # Check by filename first (fast path)
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            content = f.read()
        existing = lookup_paper(content, output_dir)
        if existing:
            return os.path.join(output_dir, existing.filename)
        register_paper(
            content,
            source_url=url,
            source_type=source_type,
            registry_dir=output_dir,
            filename=filename,
            file_size=len(content),
        )
        return output_path

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            # Handle arxiv direct PDF links
            if "arxiv.org/pdf/" in url:
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                if response and response.headers.get("content-type", "").startswith(
                    "application/pdf"
                ):
                    browser.close()
                    return _save_and_register(response.body())

            # Navigate to the page
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Find PDF links on the page
            pdf_links = page.locator('a[href$=".pdf"]').all()
            for link in pdf_links:
                href = link.get_attribute("href") or ""
                if href:
                    pdf_url = page.evaluate(
                        "(href) => new URL(href, location.href).href", href
                    )
                    try:
                        download_response = page.evaluate(
                            """async (url) => {
                                const resp = await fetch(url);
                                if (!resp.ok) return null;
                                const blob = await resp.blob();
                                return { type: blob.type, size: blob.size };
                            }""",
                            pdf_url,
                        )
                        if download_response and (
                            "pdf" in str(download_response.get("type", ""))
                            or download_response.get("size", 0) > 10000
                        ):
                            with page.expect_download() as download_info:
                                link.click()
                            download = download_info.value
                            content = (
                                download.path() if hasattr(download, "path") else None
                            )
                            if content:
                                with open(download.path(), "rb") as tmpf:
                                    content = tmpf.read()
                                browser.close()
                                return _save_and_register(content)
                    except Exception:
                        continue

            # For scholar: try clicking the main result link
            if "scholar" in url:
                result_link = page.locator("h3.gs_rt a").first
                if result_link:
                    try:
                        result_link.click()
                        page.wait_for_load_state("networkidle", timeout=15000)
                        # Try PDF links on the result page
                        pdf_links = page.locator('a[href$=".pdf"]').all()
                        for link in pdf_links:
                            try:
                                with page.expect_download(
                                    timeout=15000
                                ) as download_info:
                                    link.click()
                                download = download_info.value
                                with open(download.path(), "rb") as tmpf:
                                    content = tmpf.read()
                                browser.close()
                                return _save_and_register(content)
                            except Exception:
                                continue
                    except Exception:
                        pass

            browser.close()
    except Exception:
        pass

    return None
