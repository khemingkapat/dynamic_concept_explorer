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


---

## 2026-11-16 — Issue #5: Quantitative Text Analytics & Graph Topology Evaluation

- **Branch:** `feat/evaluation-metrics`
- **Issue:** [#5](https://github.com/khemingkapat/dynamic_concept_explorer/issues/5) — `eval: compute quantitative metrics for text analytics evaluation`

### What Was Built
- **Dependencies (`pyproject.toml`, `uv.lock`):**
  - Added `rouge-score>=0.1.2` for ROUGE-1, ROUGE-2, and ROUGE-L evaluation.
- **Evaluation Module (`src/dynamic_concept_explorer/evaluation.py`):**
  - **Graph Topology (`compute_graph_topology` -> `GraphTopologyMetrics`):** Computes `node_count`, `edge_count`, `density`, `in_degree_centrality`, `out_degree_centrality`, `betweenness_centrality`, `pagerank`, undirected-view `clustering_coefficients` / `average_clustering`, and `reciprocity`, with safe defaults for empty, 1-node, and edgeless graphs.
  - **LDA Topic Coherence (`compute_topic_coherence` -> `TopicCoherenceResult`):** Computes per-topic and mean Gensim $C_v$ coherence (`processes=1`) from `TopicClusteringResult`.
  - **Summarization ROUGE (`compute_rouge_score`, `evaluate_summaries` -> `RougeEvaluationResult`):** Computes per-document and macro-averaged ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, F1) from plain text mappings, decoupled from `summarizer.py`.
- **Tests (`tests/test_evaluation.py`):**
  - Added 6 offline unit tests covering directed/undirected graph topology, bridge betweenness, degenerate graphs, $C_v$ coherence, and ROUGE-1/2/L scoring and validation.

### Verification
- `uv run ruff check` & `uv run ruff format --check` on `evaluation.py` and `tests/test_evaluation.py`: passed (`All checks passed!`, `2 files already formatted`).
- `uv run pytest -v`: 17/17 passed.
- Architectural Result Audit (`pro` model): `🟢 CLEAN`.

---

## 2026-11-16 — Issue #2: Node Overviews (Lead Extraction, Fast TextRank, On-Demand BART)

- **Branch:** `feat/summarizer-node-overviews`
- **Issue:** [#2](https://github.com/khemingkapat/dynamic_concept_explorer/issues/2) — `feat(summarizer): adapt extractive TextRank and distillation for topic nodes`

### What Was Built
- **Article Text Extraction (`src/dynamic_concept_explorer/scraper.py`):**
  - **`extract_article_text` → `ArticleText(lead, body, .full)`:** Clean prose split at the first `<h2>`. Strips citation markers (`[1]`, `[citation needed]`), math markup, edit links, and infobox paragraphs; skips the same boilerplate sections as link ranking. Works on `<section>`-wrapped and flat markup and does not mutate the parsed page. Issue #3 (LDA) can reuse `.full` as its document text.
- **Node Summaries (`src/dynamic_concept_explorer/summarizer.py`):**
  - **`summarize_node(lead, body, method=..., n_sentences=2)` → `NodeSummary(text, sentences, method, elapsed_ms)`:** `auto` (default) returns the first 2 usable lead sentences and falls back to full-body TextRank when the lead is too thin; `lead`, `textrank`, and `bart` force one strategy. `NodeSummary.method` records what actually produced the text.
  - **Faster TextRank (`rank_sentences`, `pagerank_scores`, `textrank_sentences`):** Sparse TF-IDF product for cosine similarity plus vectorised power-iteration PageRank instead of building a NetworkX graph; sentences outside 6–60 words are dropped and input is capped at 250 sentences.
  - **On-demand BART (`bart_node_summary`, `method="bart"`):** Model is loaded once per process and only when requested; only the lead is sent so input stays under 1024 tokens. If the model cannot load or fails, the result degrades to the extractive path and is labelled `"lead (bart unavailable)"`.
  - Coursework baselines (`textrank`, `lexrank_summary`, `bart_summary`) keep their behaviour; unused imports removed and file formatted with `ruff`.
- **Comparison Script (`scripts/compare_node_summaries.py`):** Prints both overviews, latency, and ROUGE-1/2 F1 of body-only TextRank against the article lead for any list of article titles.
- **Tests (`tests/test_summarizer.py`):** 16 offline tests — text extraction on both markup styles, PageRank equivalence with `nx.pagerank` (including a dangling node), agreement with the coursework TextRank's top sentence, 2-sentence overviews across four academic topics (ML, photosynthesis, French Revolution, general relativity), thin-lead fallback, fragment filtering, BART on demand with an injected model, and BART failure fallback.
- **CI (`.github/workflows/ci.yml`):** Ruff now also covers `summarizer.py` and `scripts/`.

### Lead vs Full-Body TextRank (live, 2026-11-16)

| Topic | Body sents | Lead ms | TextRank ms | Baseline nx ms | R-1 vs lead | R-2 vs lead |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Machine_learning | 381 | 0.2 | 14.2 | 212.9 | 0.216 | 0.045 |
| Photosynthesis | 282 | 0.6 | 12.9 | 54.0 | 0.103 | 0.025 |
| French_Revolution | 543 | 0.3 | 19.9 | 93.5 | 0.186 | 0.035 |
| General_relativity | 352 | 0.5 | 16.6 | 240.5 | 0.160 | 0.050 |
| Game_theory | 360 | 0.2 | 13.5 | 77.1 | 0.158 | 0.009 |
| Natural_language_processing | 42 | 0.1 | 3.3 | 5.5 | 0.339 | 0.069 |

- **Decision: lead extraction is the default.** Lead sentences are definitions ("X is a ..."), which is what a node popover needs; body TextRank picks central but mid-article detail sentences (ROUGE-2 against the lead ≤ 0.07) and breaks on formula-heavy articles such as Game_theory, where stripped math leaves gaps. TextRank is kept as the fallback for pages without a usable lead.
- **Latency caveat:** the "Baseline nx" column ranks every body sentence while the new path ranks at most 250 filtered sentences, so the speed-up combines the sentence cap and the vectorised PageRank; it is not a like-for-like measure of PageRank alone.
- **ROUGE caveat:** the lead is used as the reference, so these scores measure how well body TextRank recovers the abstract; they are not a fair score for the lead strategy itself.

### Verification
- `uv run ruff check` & `uv run ruff format --check` on `scraper.py`, `summarizer.py`, `tests/`, `scripts/`: passed.
- `uv run pytest`: 22/22 passed (6 scraper + 16 summarizer).
- `uv run python verify_baseline.py`: passed.
- Not run: the real `facebook/bart-large-cnn` model (about 1.6 GB download); the BART path is covered only with an injected stand-in.

### Follow-Up / Backlog
- Wire `extract_article_text` + `summarize_node` into the node payload for the UI (issue #4).
- Reuse `rouge_n_f1` from the comparison script, or a ROUGE library, for the formal evaluation (issue #5).
- Formula-heavy articles lose inline math in body sentences; consider keeping `alttext` if TextRank output is ever shown.
