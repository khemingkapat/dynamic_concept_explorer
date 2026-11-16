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

---

## 2026-11-16 — Issue #3: Unsupervised LDA Clustering for Topic Nodes

- **Branch:** `feat/lda-topic-clustering`
- **Issue:** [#3](https://github.com/khemingkapat/dynamic_concept_explorer/issues/3) — `feat(modeling): cluster discovered topic nodes using unsupervised LDA`

### What Was Built
- **Scraper Document Extraction (`src/dynamic_concept_explorer/scraper.py`):**
  - **`extract_article_text`:** Extracts clean `<p>` prose from `div#bodyContent` while skipping boilerplate sections (`see also`, `references`, `external links`, `notes`, `further reading`, `bibliography`, `sources`).
  - **`fetch_neighborhood_documents`:** Fetches cached HTML for `seed_url` + active `candidates` (ignoring `pruned` nodes) and returns `dict[canonical_href, str]` keyed identically to `NeighborhoodResult.edges`.
  - **`_resolve_fetcher`:** Shared helper for cached page fetching across `discover_neighborhood` and `fetch_neighborhood_documents`.
- **Unsupervised LDA Clustering (`src/dynamic_concept_explorer/topic_modeling.py`):**
  - **Structured Outputs:** Added frozen dataclasses `TopicDescriptor`, `NodeTopicAssignment`, and `TopicClusteringResult` (exposing `topics`, `assignments`, `model`, `dictionary`, `corpus`, and `tokenized_docs` for Issue #4 UI and Issue #5 $C_v$ coherence).
  - **Tunable Hyperparameters:** Extended `preprocess_text(..., extra_stopwords=None)` and `train_lda(..., alpha="symmetric", eta=None)`, and added `cluster_topic_nodes` exposing `num_topics` (default `4`), `no_below` (default `2`), `no_above` (default `0.85`), `alpha`, `eta`, and `extra_stopwords` while keeping `passes=10` fixed internally.
  - **Edge-Case Guards:** Fallback to `no_below=1, no_above=1.0` if `filter_extremes` empties the dictionary, `ValueError` on empty/stopword-only corpora, and uniform-distribution fallback (`confidence=0.0`) for empty per-node BoWs.
- **Tests (`tests/test_topic_modeling.py`):**
  - Added 5 offline unit tests covering `preprocess_text` + `extra_stopwords`, `build_corpus_and_dictionary` filtering, multi-domain LDA clustering, edge-case guards, and `fetch_neighborhood_documents` integration.

### Verification
- `uv run ruff check` & `uv run ruff format --check` on touched files: passed (`All checks passed!`, `4 files already formatted`).
- `uv run pytest -v`: 11/11 passed.
- `uv run python verify_baseline.py`: passed.
- Live cached check (`Machine_learning`, 21 documents, `num_topics=4`, `extra_stopwords={'machine', 'learning'}`): produced 4 distinct topics (e.g., `/wiki/Deep_learning` → `network, train, neural` at `1.00`; `/wiki/Bayesian_network` → `variable, rule, game` at `1.00`).
