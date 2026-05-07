# AGENTS.md

## Rules

- NEVER disclose environment variables or secrets.
- Always commit and push completed work before ending a session.
- Always pull from the remote before starting new work.

## Development Environment

This project uses **uv** for Python dependency management and **nix** (via `flake.nix`) for reproducible development shells.

### Adding Dependencies

```bash
uv add <package-name>
```

After adding new dependencies, you **must** run `direnv allow` (or `direnv reload`) for the nix shell to pick up the changes:

```bash
direnv allow
```

The agent cannot use newly added packages until direnv has been reloaded. When the agent requests a new package, add it with `uv add`, then tell the agent to proceed — it will work after the shell reloads.

### Formatting

```bash
treefmt
```

Formatters: `black` (Python), `nixfmt` (Nix), `mdformat` (Markdown), `taplo` (TOML).

## Git Workflow

**At the start of every session:**
```bash
git pull origin main
```

**After completing any meaningful unit of work:**
```bash
git add <files>
git commit -m "Descriptive message"
git push
```

**If a push is rejected:**
```bash
git pull --rebase
git push
```

## Project Architecture

```
paperchecker/
├── src/paperchecker/
│   ├── __init__.py
│   ├── main.py          # CLI entry point (typer)
│   ├── config.py        # Environment config (API keys, defaults)
│   ├── llm.py           # LLM backends (DeepSeek, OpenAI, Claude)
│   ├── puller.py        # Citation puller - reads LaTeX, extracts citations
│   ├── downloader.py    # Paper downloader using playwright
│   ├── extractor.py     # Text extraction from PDFs/txt
│   ├── phraser.py       # Spacy phrase splitting into numbered phrases
│   ├── checker.py       # LLM-powered citation verification (1-5 scale)
│   ├── manifest.py      # Manifest management (MD files)
│   └── registry.py      # Hash registry for papers and check cache
├── pyproject.toml
├── README.md
├── LICENSE
├── flake.nix
└── AGENTS.md
```

### Pipeline

1. **puller** — Reads LaTeX `\cite{}` and `\footnote{}`, extracts claims and citation keys
2. **downloader** — Uses playwright to fetch papers from known repositories (arXiv, Google Scholar, DOI resolvers)
3. **extractor** — Converts PDFs to plaintext (pypdf → pdfplumber → pdftotext fallback chain)
4. **phraser** — Splits source text into numbered phrases with spacy
5. **checker** — Sends claim + numbered phrases to LLM, gets 1-5 confidence score
6. **manifest** — Stores paper location, extracted text location, and verification status as MD
7. **registry** — Content-hash based deduplication: avoids re-downloading papers and re-checking citations, store in `_check_cache.json`

### LLM Backends

Configured via environment variables. The checker uses the first available backend:

| Backend | Environment Variable | Model |
|---------|---------------------|-------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` (1M context) |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |

Semantic Scholar API key (optional, avoids rate limits):
| `SEMANTIC_SCHOLAR_API_KEY` | Free from https://api.semanticscholar.org/ |

### Confidence Scale (1-5)

| Score | Meaning |
|-------|---------|
| 1 | No content suggesting the claim |
| 2 | Some related information but insufficient for citation |
| 3 | Related but not strongly supported |
| 4 | Supported with some question of interpretation |
| 5 | Exact match of the cited content |

### Manifest Format

Manifests are Markdown files stored in a `manifests/` directory. Each manifest tracks one paper being checked:

```markdown
# Paper: <title>

**Source**: `<path to LaTeX/PDF>`
**Extracted Text**: `<path to extracted plaintext>`
**Status**: `in_progress | complete`

## Citations

| # | Claim | Citation Key | Source Paper | Status | Confidence | Phrase |
|---|-------|-------------|-------------|--------|------------|--------|
| 1 | ... | ... | ... | verified | 4 | [23] |
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `DEEPSEEK_API_KEY` | No | DeepSeek API key |
| `OPENAI_API_KEY` | No | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | Anthropic API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Semantic Scholar API key (free, avoids rate limits) |

At least one LLM API key must be set for verification to work.

## Key Reference Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | This file — workflow and architecture |
| `pyproject.toml` | Package metadata and dependencies |
| `flake.nix` | Nix development shell |
| `LICENSE` | Apache 2.0 (Scidonia Limited) |
