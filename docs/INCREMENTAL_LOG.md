# Incremental Development Log

## 2026-11-13 — Issue #1: Scraper Outbound Ranking, 2-Pass Pruning & Caching

- **Branch:** `feat/scraper-ranking-indegree`
- **Issue:** [#1](https://github.com/khemingkapat/dynamic_concept_explorer/issues/1) — `feat(scraper): implement outbound candidate ranking and in-degree filtering`

### What Was Built
- **Tooling & Standards (`pyproject.toml`, `.gitignore`):**
  - Added `ruff` and `pytest` to `[dependency-groups] dev` with `[tool.ruff]` (`line-length = 100`, `target-version = "py311"`) and `[tool.pytest.ini_options]`.
  - Gitignored `.cache/`, `.pytest_cache/`, and `.ruff_cache/`.
- **Scraper Pipeline (`src/dynamic_concept_explorer/scraper.py`):**
  - **Caching (`fetch_wikipedia_page`, `clear_page_cache`):** Two-tier in-memory + persistent JSON disk cache in `.cache/wiki/` (default 7-day TTL) to support pre-cached demos and fast 2-pass crawls.
  - **Canonical Extraction (`extract_link_stats`, `_top_level_sections`, `_canonical_wiki_href`):** Normalizes relative `/wiki/...` and absolute `https://*.wikipedia.org/wiki/...` hrefs, strips `#fragment` and query strings, skips boilerplate sections (`see also`, `references`, `external links`, `notes`, `further reading`, `bibliography`, `sources`), and avoids double-counting nested `<section>` tags.
  - **Injectable Outbound Ranking (`default_link_scorer`, `rank_outbound_links`):** Default scorer `2.0 * int(in_lead) + log1p(count)` passed via dependency injection (`LinkScorer`), bounded by `top_k=20`.
  - **2-Pass Neighborhood Pruning (`discover_neighborhood`):** Computes neighborhood in-degree across top-k candidates, prunes nodes below `min_in_degree=2` with a `min_nodes=5` fallback, and returns `NeighborhoodResult` (`seed_url`, `candidates`, `pruned`, `edges`).
- **Tests (`tests/test_scraper.py`):**
  - Added 6 offline unit tests covering boilerplate/canonicalization, absolute URLs + nested sections, default/custom scoring, 2-pass pruning + fallback, fetch error handling, and memory/disk caching.

### Verification
- `uv run ruff check` & `uv run ruff format --check`: passed (`All checks passed!`, `2 files already formatted`).
- `uv run pytest -v`: 6/6 passed.
- `uv run python verify_baseline.py`: passed.
- Live check (`Machine_learning`, `top_k=20`, `min_in_degree=2`): 20 candidates kept, 21 edge sources, cached to `.cache/wiki/`.

### Follow-Up / Backlog (from Architectural Audit)
- Bound `_MEMORY_CACHE` with LRU eviction and key by `(cache_dir, url)`.
- Replace `":" in canonical` with explicit Wikipedia namespace prefix matching so titles with colons are preserved.
- Add optional polite rate-limiting/backoff for uncached Pass-2 requests.
