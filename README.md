# Paperchecker

LLM-powered citation verifier for academic papers. Extracts citations from LaTeX, locates and downloads source papers (arXiv, Semantic Scholar, Sci-Hub), and verifies that each claim is substantiated by the cited text.

## Quick Start

```bash
# Install
uv sync
direnv allow

# Check a paper
paperchecker check paper.tex --papers-dir papers --manifests-dir manifests
```

## Commands

### `check` — Run the full verification pipeline

```bash
paperchecker check paper.tex
```

Extracts citations, resolves source files, downloads missing papers, and verifies each claim against the source text using an LLM (1–5 confidence scale).

Options:
- `--papers-dir` — directory containing source PDFs (default: `papers/`)
- `--manifests-dir` — output directory for manifests (default: `manifests/`)
- `--backend` — LLM to use: `deepseek`, `openai`, or `claude` (auto-detected from environment)
- `--no-auto-download` — skip online search for missing papers

### `pull` — Extract citations and build a manifest

```bash
paperchecker pull paper.tex
```

Reads `\cite{}` and `\footnote{}` commands, resolves `\input{}`/`\include{}` across multi-file projects, and generates a Markdown manifest.

### `download` — Download a paper by arXiv ID, DOI, or search

```bash
paperchecker download 2401.12345 --source-type arxiv
paperchecker download 10.1145/5956.5957 --source-type doi
paperchecker download "Turchin concept supercompiler" --source-type search
```

### `register` — Register existing PDFs in the hash registry

```bash
paperchecker register papers/
```

Scans a directory of PDFs and registers them by content hash so they won't be re-downloaded.

### `extract` — Extract plaintext from a PDF

```bash
paperchecker extract paper.pdf --output-file paper.txt
```

### `phrase` — Split a text file into numbered phrases

```bash
paperchecker phrase paper.txt
```

### `list-backends` — Show available LLM backends

```bash
paperchecker list-backends
```

## Download Sources

Papers are located and downloaded via a fallback chain:

1. **Local files** — papers already in the `--papers-dir`
2. **arXiv** — API search with title similarity and year matching
3. **Semantic Scholar** — free API, covers all years and disciplines
4. **Unpaywall** — locates legitimate open-access copies by DOI
5. **Sci-Hub** — resolves DOI paywalls

Downloaded papers are registered by content hash (SHA-256) so identical papers are never downloaded twice.

## LLM Backends

Set at least one of these environment variables:

| Backend   | Variable            | Model                          |
|-----------|---------------------|--------------------------------|
| DeepSeek  | `DEEPSEEK_API_KEY`  | `deepseek-chat` (1M context)   |
| OpenAI    | `OPENAI_API_KEY`    | `gpt-4o`                       |
| Claude    | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514`     |

DeepSeek is preferred for its 1M token context window, which allows checking claims against entire papers.

## Verification Scale

| Score | Meaning                                          |
|-------|--------------------------------------------------|
| 1     | No content suggesting the claim                  |
| 2     | Some related information but insufficient        |
| 3     | Related but not strongly supported               |
| 4     | Supported with some question of interpretation   |
| 5     | Exact match of the cited content                 |

## Manifest Format

Each paper gets a Markdown manifest tracking verification state:

```markdown
# Paper: sample

**Source**: `paper.tex`
**Extracted Text**: `papers/mills1956.pdf`
**Status**: `complete`

## Citations

| # | Claim | Citation Key | Source Paper | Status | Confidence | Phrase |
|---|-------|-------------|-------------|--------|------------|--------|
| 1 | ...   | mills1956   |             | verified | 5 | [29] |
```

## Project Structure

```
paperchecker/
├── src/paperchecker/
│   ├── main.py          # CLI entry point (typer)
│   ├── config.py        # Environment config (API keys)
│   ├── llm.py           # LLM backends (DeepSeek, OpenAI, Claude)
│   ├── puller.py        # LaTeX citation extraction + bibtex resolution
│   ├── downloader.py    # Multi-source paper downloader
│   ├── extractor.py     # PDF/text extraction
│   ├── phraser.py       # Spacy phrase splitting
│   ├── checker.py       # LLM citation verification
│   ├── manifest.py      # Manifest management
│   └── registry.py      # Content-hash deduplication
└── pyproject.toml
```

## License

Apache 2.0 — Scidonia Limited
