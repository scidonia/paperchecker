"""CLI entry point — typer-based command-line interface for paperchecker."""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from paperchecker.config import Config
from paperchecker.puller import (
    extract_citations,
    load_cache,
    save_cache,
    resolve_source,
    get_citation_info,
    build_search_query,
)
from paperchecker.extractor import extract_text
from paperchecker.phraser import split_phrases, format_numbered_phrases
from paperchecker.checker import verify_claim, guess_doi
from paperchecker.manifest import (
    create_manifest,
    add_citation,
    update_citation,
    write_manifest,
)
from paperchecker.downloader import (
    download_from_arxiv,
    download_from_doi,
    download_from_scholar,
    search_arxiv,
    search_semantic_scholar,
    search_crossref,
    _download_via_libgen,
)
from paperchecker.registry import (
    check_citation_hash,
    lookup_check,
    store_check,
    register_paper,
)

app = typer.Typer(
    name="paperchecker",
    help="LLM-powered citation verifier for academic papers.",
)
console = Console()


@app.command()
def pull(
    tex_file: str = typer.Argument(..., help="Path to the LaTeX .tex file"),
    out_dir: str = typer.Option("manifests", help="Output directory for manifest"),
):
    """Extract citations from a LaTeX file and build a manifest."""
    tex_path = Path(tex_file)
    if not tex_path.exists():
        console.print(f"[red]File not found: {tex_file}[/red]")
        raise typer.Exit(1)

    citations = extract_citations(str(tex_path))

    manifest = create_manifest(
        paper_title=tex_path.stem,
        source_path=str(tex_path),
        manifest_dir=out_dir,
    )

    for cit in citations:
        add_citation(manifest, cit.claim, cit.citation_key)

    path = write_manifest(manifest)

    console.print(f"[green]Found {len(citations)} citations[/green]")
    console.print(f"[green]Manifest written to: {path}[/green]")


@app.command()
def check(
    tex_file: str = typer.Argument(..., help="Path to the LaTeX .tex file"),
    papers_dir: str = typer.Option(
        "_paperchecker/papers", help="Directory containing source PDFs"
    ),
    manifests_dir: str = typer.Option(
        "_paperchecker", help="Manifest directory"
    ),
    backend: Optional[str] = typer.Option(
        None, help="LLM backend to use (deepseek, openai, claude)"
    ),
    no_auto_download: bool = typer.Option(
        False, "--no-auto-download", help="Skip online search for missing papers"
    ),
):
    """Run the full citation verification pipeline."""
    config = Config(papers_dir=papers_dir, manifests_dir=manifests_dir)

    if not config.preferred_backend:
        console.print(
            "[red]No LLM API key configured. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.[/red]"
        )
        raise typer.Exit(1)

    if backend and backend not in config.available_backends:
        console.print(
            f"[red]Backend '{backend}' not available (no API key). Available: {config.available_backends}[/red]"
        )
        raise typer.Exit(1)

    tex_path = Path(tex_file)
    if not tex_path.exists():
        console.print(f"[red]File not found: {tex_file}[/red]")
        raise typer.Exit(1)

    console.print(f"Using backend: [bold]{backend or config.preferred_backend}[/bold]")

    # Step 1: Extract citations
    console.print("\n[bold]Step 1: Extracting citations...[/bold]")
    citations = extract_citations(str(tex_path))
    console.print(f"  Found {len(citations)} citations")

    # Step 2: Build manifest
    manifest = create_manifest(
        paper_title=tex_path.stem,
        source_path=str(tex_path),
        manifest_dir=manifests_dir,
    )

    # Load cache
    cache = load_cache(manifests_dir, str(tex_path))

    # Step 3: Verify each citation
    console.print("\n[bold]Step 2: Verifying citations...[/bold]")

    results_table = Table(title="Verification Results")
    results_table.add_column("#", style="dim")
    results_table.add_column("Citation", style="cyan")
    results_table.add_column("Confidence")
    results_table.add_column("Status")

    verified = 0
    unsubstantiated = 0
    unchecked = 0

    for i, cit in enumerate(citations):
        entry = add_citation(manifest, cit.claim, cit.citation_key)

        cached = None

        # Check per-tex-file cache first (fast, backward compat)
        if cit.cache_key in cache:
            cached = cache[cit.cache_key]

        if cached:
            update_citation(
                entry, cached["status"], cached["confidence"], cached.get("phrase")
            )
            if cached["status"] == "verified":
                status_icon = "✓"
                confidence = cached["confidence"]
                verified += 1
            elif cached["status"] == "unchecked":
                status_icon = "?"
                confidence = 0
                unchecked += 1
            else:
                status_icon = "✗"
                confidence = cached["confidence"]
                unsubstantiated += 1
            console.print(
                f"  [dim]{cit.citation_key[:50]}... (cached)[/dim]"
            )
        else:
            # Try to find and verify the paper
            source_path = resolve_source(
                cit.citation_key, papers_dir, tex_path=str(tex_path)
            )

            # If not found locally, try to search and download online
            if not source_path and not no_auto_download:
                query = build_search_query(cit.citation_key, str(tex_path))
                if query:
                    info = get_citation_info(cit.citation_key, str(tex_path))
                    expected = info.get("title")
                    year = info.get("year")
                    console.print(
                        f"  [yellow]Searching: {query[:60]}...[/yellow]"
                    )
                    source_path = search_arxiv(
                        query,
                        papers_dir,
                        expected_title=expected,
                        expected_year=year,
                    )
                    if not source_path:
                        source_path = search_semantic_scholar(
                            query,
                            papers_dir,
                            expected_title=expected,
                            expected_year=year,
                        )
                    if not source_path:
                        source_path = search_crossref(
                            query,
                            papers_dir,
                            expected_title=expected,
                            expected_year=year,
                        )
                    # Final fallback: ask LLM for the DOI, try sci-hub
                    if (
                        not source_path
                        and info.get("author")
                        and info.get("title")
                    ):
                        guessed_doi = guess_doi(
                            config,
                            info["title"],
                            info.get("author", ""),
                            info.get("year", ""),
                        )
                        if guessed_doi:
                            console.print(
                                f"  [dim]LLM guessed DOI: {guessed_doi}[/dim]"
                            )
                            source_path = download_from_doi(
                                guessed_doi, papers_dir
                            )
                    # Last resort: try libgen for books and hard-to-find papers
                    if (
                        not source_path
                        and info.get("title")
                        and info.get("author")
                    ):
                        source_path = _download_via_libgen(
                            info["title"],
                            info["author"],
                            papers_dir,
                        )
                    if source_path:
                        console.print(
                            f"  [green]Downloaded: {os.path.basename(source_path)}[/green]"
                        )

            if source_path:
                source_text = extract_text(source_path)
                if source_text:
                    manifest.extracted_text_path = source_path

                    # Check content-aware global cache (claim + citation + source hash)
                    check_key = check_citation_hash(
                        cit.claim, cit.citation_key, source_text
                    )
                    global_cached = lookup_check(check_key, manifests_dir)

                    if global_cached:
                        update_citation(
                            entry,
                            global_cached["status"],
                            global_cached["confidence"],
                            global_cached.get("phrase"),
                        )
                        status_icon = (
                            "✓" if global_cached["status"] == "verified" else "✗"
                        )
                        confidence = global_cached["confidence"]
                        if global_cached["status"] == "verified":
                            verified += 1
                        else:
                            unsubstantiated += 1
                        console.print(
                            f"  [dim]{cit.citation_key[:50]}... (cached)[/dim]"
                        )
                    else:
                        result = verify_claim(
                            config,
                            cit.claim,
                            source_text,
                            os.path.basename(source_path),
                            backend=backend,
                        )
                        if result:
                            update_citation(
                                entry,
                                result.status,
                                result.confidence,
                                f"[{result.phrase_index}]",
                            )
                            status_icon = "✓" if result.evidence_found else "✗"
                            confidence = result.confidence
                            if result.evidence_found:
                                verified += 1
                            else:
                                unsubstantiated += 1

                            result_data = {
                                "status": result.status,
                                "confidence": result.confidence,
                                "phrase": f"[{result.phrase_index}]",
                                "reason": result.reason,
                            }
                            cache[cit.cache_key] = result_data
                            store_check(
                                check_key,
                                result.status,
                                result.confidence,
                                result.reason,
                                f"[{result.phrase_index}]",
                                manifests_dir,
                            )
                        else:
                            status_icon = "?"
                            confidence = 0
                            unchecked += 1
                            cache[cit.cache_key] = {
                                "status": "unchecked",
                                "confidence": 0,
                                "phrase": "",
                                "reason": "LLM verification failed",
                            }
                else:
                    status_icon = "?"
                    confidence = 0
                    unchecked += 1
                    cache[cit.cache_key] = {
                        "status": "unchecked",
                        "confidence": 0,
                        "phrase": "",
                        "reason": "Text extraction failed",
                    }
            else:
                status_icon = "?"
                confidence = 0
                unchecked += 1
                cache[cit.cache_key] = {
                    "status": "unchecked",
                    "confidence": 0,
                    "phrase": "",
                    "reason": "No source found",
                }

        results_table.add_row(
            str(i + 1),
            cit.citation_key[:60],
            str(confidence),
            status_icon,
        )

    manifest.status = "complete"
    write_manifest(manifest)
    save_cache(manifests_dir, str(tex_path), cache)

    console.print(results_table)
    console.print(
        f"\n[bold]Summary:[/bold] {verified} verified, {unsubstantiated} unsubstantiated, {unchecked} unchecked"
    )

    if unchecked > 0:
        console.print(
            "[yellow]Some citations could not be checked. Run 'paperchecker download' first.[/yellow]"
        )


@app.command()
def download(
    source: str = typer.Argument(..., help="arXiv ID, DOI, or search query"),
    source_type: str = typer.Option(
        "auto", help="Source type: arxiv, doi, scholar, or auto"
    ),
    output_dir: str = typer.Option(
        "papers", help="Output directory for downloaded PDFs"
    ),
    headless: bool = typer.Option(True, help="Run browser headless"),
):
    """Download a paper PDF from arXiv, DOI resolver, or Google Scholar."""
    console.print(f"Downloading from {source_type}: {source}")

    if source_type == "arxiv" or (source_type == "auto" and _looks_like_arxiv(source)):
        path = download_from_arxiv(source, output_dir, headless)
    elif source_type == "doi" or (source_type == "auto" and _looks_like_doi(source)):
        path = download_from_doi(source, output_dir, headless)
    elif source_type == "search" or source_type == "auto":
        path = search_arxiv(source, output_dir, headless)
    else:
        path = download_from_scholar(source, output_dir, headless)

    if path:
        console.print(f"[green]Downloaded: {path}[/green]")
    else:
        console.print(f"[red]Failed to download: {source}[/red]")
        raise typer.Exit(1)


@app.command()
def phrase(
    text_file: str = typer.Argument(
        ..., help="Path to a .txt file to split into phrases"
    ),
    max_phrases: int = typer.Option(500, help="Maximum number of phrases"),
):
    """Split a text file into numbered phrases."""
    with open(text_file) as f:
        text = f.read()

    phrases = split_phrases(text, max_phrases=max_phrases)
    output = format_numbered_phrases(phrases, truncate=None)

    console.print(output)
    console.print(f"\n[dim]{len(phrases)} phrases[/dim]")


@app.command()
def extract(
    pdf_file: str = typer.Argument(..., help="Path to a PDF or .txt file"),
    output_file: Optional[str] = typer.Option(
        None, help="Output .txt file (prints to stdout if not set)"
    ),
):
    """Extract plaintext from a PDF."""
    text = extract_text(pdf_file)
    if text is None:
        console.print("[red]Failed to extract text[/red]")
        raise typer.Exit(1)

    if output_file:
        with open(output_file, "w") as f:
            f.write(text)
        console.print(f"[green]Extracted to: {output_file}[/green]")
    else:
        console.print(text[:5000])
        if len(text) > 5000:
            console.print(f"\n[dim]... ({len(text)} chars total)[/dim]")


@app.command()
def list_backends():
    """List available LLM backends."""
    config = Config()
    backends = config.available_backends
    if not backends:
        console.print("[red]No LLM backends configured.[/red]")
        console.print("Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")
    else:
        for b in backends:
            preferred = " (preferred)" if b == config.preferred_backend else ""
            console.print(f"  - {b}{preferred}")


@app.command()
def register(
    directory: str = typer.Argument(..., help="Directory of PDFs to register"),
):
    """Register existing PDFs in the hash registry to avoid re-downloading."""
    from paperchecker.registry import lookup_paper

    pdfs = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    if not pdfs:
        console.print("[yellow]No PDFs found in directory[/yellow]")
        return

    registered = 0
    skipped = 0
    for filename in pdfs:
        filepath = os.path.join(directory, filename)
        with open(filepath, "rb") as f:
            content = f.read()
        existing = lookup_paper(content, directory)
        if existing:
            skipped += 1
            console.print(f"  [dim]already registered:[/dim] {filename}")
        else:
            register_paper(
                content,
                source_url=f"file://{filepath}",
                source_type="local",
                registry_dir=directory,
                filename=filename,
                file_size=len(content),
            )
            registered += 1
            console.print(f"  [green]registered:[/green] {filename}")

    console.print(
        f"\n[bold]{registered} registered, {skipped} already known[/bold]"
    )


@app.command()
def scan(
    papers_dir: str = typer.Option(
        "_paperchecker/papers", help="Directory of PDFs to scan"
    ),
    manifests_dir: str = typer.Option(
        "_paperchecker", help="Directory for scan manifest"
    ),
):
    """Scan PDFs with LLM to extract metadata, match against citations.

    Reads the first few pages of each unregistered PDF, asks the LLM for
    title/author/year, and stores the metadata for citation matching.
    Useful for papers downloaded via sci-hub or other sources that don't
    have immediate bibtex associations.
    """
    from paperchecker.extractor import extract_text
    from paperchecker.llm import call_llm

    if not os.path.isdir(papers_dir):
        console.print(f"[red]Papers directory not found: {papers_dir}[/red]")
        raise typer.Exit(1)

    pdfs = sorted(
        [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    )

    scans_path = os.path.join(manifests_dir, "_scans.json")
    scans = {}
    if os.path.exists(scans_path):
        with open(scans_path) as f:
            scans = json.load(f)

    new = 0
    for filename in pdfs:
        if filename in scans:
            continue

        filepath = os.path.join(papers_dir, filename)
        text = extract_text(filepath, max_chars=4000)
        if not text:
            continue

        prompt = f"""Extract the bibliographic metadata from this academic paper text.
Return ONLY a JSON object with these fields:
{{"title": "...", "author": "Lastname", "year": "YYYY"}}
If you cannot determine a field, use "unknown".

Paper text:
{text[:3000]}

JSON:"""

        resp = call_llm(config, prompt)
        if not resp:
            continue

        # Parse JSON from response
        try:
            # Extract JSON object
            import re
            m = re.search(r'\{[^}]+\}', resp)
            if m:
                meta = json.loads(m.group(0))
                scans[filename] = {
                    "title": meta.get("title", "unknown"),
                    "author": meta.get("author", "unknown"),
                    "year": str(meta.get("year", "unknown")),
                }
                new += 1
                console.print(
                    f"  [green]{filename}[/green] → "
                    f"{meta.get('author', '?')}, "
                    f"{meta.get('year', '?')}"
                )
                with open(scans_path, "w") as f:
                    json.dump(scans, f, indent=2)
        except Exception:
            pass

    console.print(f"\n[bold]{new} new scans, {len(scans)} total[/bold]")


def _looks_like_arxiv(s: str) -> bool:
    return bool(re.match(r"\d{4}\.\d+", s))


def _looks_like_doi(s: str) -> bool:
    return "/" in s and ("10." in s or "doi.org" in s)
