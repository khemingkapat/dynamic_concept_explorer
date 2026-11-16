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

## 2026-11-16 — Issue #2: Node Overviews with the README Methods (TextRank, LexRank, BART)

- **Branch:** `feat/summarizer-node-overviews`
- **Issue:** [#2](https://github.com/khemingkapat/dynamic_concept_explorer/issues/2) — `feat(summarizer): adapt extractive TextRank and distillation for topic nodes`
- **Status:** work in progress.

### What Was Built
- **Article Text Extraction (`src/dynamic_concept_explorer/scraper.py`):**
  - **`extract_article_text` → `ArticleText(lead, body, .full)`:** Clean prose split at the first `<h2>`. Strips citation markers (`[1]`, `[citation needed]`), math markup, edit links, and infobox paragraphs; skips the same boilerplate sections as link ranking. Works on `<section>`-wrapped and flat markup and does not mutate the parsed page. Issue #3 (LDA) can reuse `.full` as its document text.
- **Node Summaries (`src/dynamic_concept_explorer/summarizer.py`):** only the three methods from README §2.2 are used.
  - **`summarize_node(lead, body, method=..., scope=..., n_sentences=2)` → `NodeSummary(text, sentences, method, scope, elapsed_ms)`.**
  - **`method`:** `textrank` (default), `lexrank` (optional `background` documents for IDF), `bart` (on demand).
  - **`scope`:** `lead` (default) ranks the lead section, following the roadmap's "TextRank lead-section scoring"; it switches to the full article when the lead has fewer than `n_sentences` usable sentences. `full` ranks lead + body.
  - **TextRank (`rank_sentences`, `textrank_sentences`):** README steps 1–5 unchanged in method (`sent_tokenize`, TF-IDF, zero-diagonal cosine matrix, `networkx` graph, `nx.pagerank`, top-k in original order). Graph construction uses the sparse product `X @ X.T` on L2-normalised TF-IDF rows instead of dense pairwise cosine. For nodes, `k` is fixed at 2 sentences instead of the ratio `r = 0.28`.
  - **LexRank (`lexrank_sentences`):** `lexrank` library with cosine threshold 0.25.
  - **BART (`bart_node_summary`):** `facebook/bart-large-cnn`, loaded once per process and only when requested. Output is trimmed to `n_sentences` complete sentences: a last sentence cut off by the token limit is dropped. If it cannot load or fails, the result falls back to TextRank and is labelled `"textrank (bart unavailable)"`.
  - **Pre-filter for all extractive methods:** sentences outside 6–60 words are dropped and input is capped at 250 sentences. This is an addition not described in the README.
  - Coursework baselines (`textrank`, `lexrank_summary`, `bart_summary`) keep their behaviour; unused imports removed and file formatted with `ruff`.
- **Comparison Script (`scripts/compare_node_summaries.py`):** Prints the overview, latency, and ROUGE-1/2 F1 against the article lead for TextRank-lead, TextRank-full, LexRank-full, and (with `--bart`) BART.
- **Tests (`tests/test_summarizer.py`):** 21 offline tests — text extraction on both markup styles, agreement with the coursework TextRank's top sentence, default TextRank-over-lead and full-scope TextRank/LexRank across four academic topics (ML, photosynthesis, French Revolution, general relativity), LexRank with a background corpus, thin-lead switch to the full article, fragment filtering, BART on demand with an injected model, BART output trimming, BART failure fallback, and rejection of unknown methods/scopes.
- **CI (`.github/workflows/ci.yml`):** Ruff now also covers `summarizer.py` and `scripts/`.

### Lead-Section vs Full-Article (live, 2026-11-16)

ROUGE is F1 against the article lead, reported only for full-article methods (TextRank-lead selects from the lead itself, so scoring it against the lead would be circular).

| Topic | Sents | TR-lead ms | TR-full ms | TR-full R-1 | TR-full R-2 | LR-full ms | LR-full R-1 | LR-full R-2 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Machine_learning | 388 | 20.2 | 131.7 | 0.216 | 0.045 | 307.6 | 0.122 | 0.009 |
| Photosynthesis | 306 | 4.4 | 36.7 | 0.089 | 0.032 | 285.0 | 0.110 | 0.018 |
| French_Revolution | 556 | 3.5 | 35.0 | 0.262 | 0.179 | 294.4 | 0.143 | 0.025 |
| General_relativity | 372 | 4.1 | 37.7 | 0.301 | 0.299 | 304.3 | 0.098 | 0.018 |
| Game_theory | 366 | 3.1 | 42.3 | 0.158 | 0.009 | 278.8 | 0.103 | 0.010 |
| Natural_language_processing | 46 | 3.3 | 5.2 | 0.695 | 0.667 | 49.2 | 0.243 | 0.014 |

- **Default: TextRank over the lead section.** In all six topics it returned the defining first sentence plus one other lead sentence, in 3–20 ms (the 20 ms on the first topic includes first-call warm-up).
- **TextRank over the full article** picked a lead sentence in three of six topics but returned mid-article detail in the others, and unreadable text for Game_theory, where stripped formulas leave gaps.
- **LexRank over the full article** never picked a lead sentence (ROUGE-2 ≤ 0.025) and was the slowest extractive option (about 280–310 ms on long articles).
- Timings are single runs on one laptop.

### BART on the Lead Section (real `facebook/bart-large-cnn`, CPU, 2026-11-16)

| Topic | BART ms | Opens with the definition |
| :--- | ---: | :--- |
| Machine_learning | 12806 | yes |
| Photosynthesis | 14134 | yes |
| French_Revolution | 6061 | no |
| General_relativity | 9428 | no |
| Game_theory | 4998 | yes |
| Natural_language_processing | 3404 | yes |

- **Latency:** 3.4–14.1 s per node on CPU, against 3–19 ms for TextRank on the lead. The first row includes loading the cached model from disk; the very first run also downloaded 1.6 GB (80 s). A 20-node graph would take minutes, so BART stays on demand and is not the default.
- **Quality:** fluent and shorter than the extractive output, but in two of six topics (French_Revolution, General_relativity) the summary skipped the defining sentence and opened with a secondary point.
- **Fixed after the first real run:** BART returned three sentences in four of six topics and cut the Photosynthesis summary mid-sentence ("…the oxygen content of the Earth's") at `max_new_tokens=80`. Output is now trimmed to two complete sentences; the rerun above returned exactly two complete sentences for all six topics.

### Verification
- `uv run ruff check` & `uv run ruff format --check` on `scraper.py`, `summarizer.py`, `tests/`, `scripts/`: passed.
- `uv run pytest`: 27/27 passed (6 scraper + 21 summarizer).
- `uv run python verify_baseline.py`: passed.
- `uv run python scripts/compare_node_summaries.py --bart`: ran end to end with the real model on all six topics. Unit tests still use an injected stand-in so CI does not download the model.

### Follow-Up / Backlog
- Merge `main` into this branch (PR #8, LDA topic clustering, was merged after the branch was cut).
- Wire `extract_article_text` + `summarize_node` into the node payload for the UI (issue #4).
- Reuse `rouge_n_f1` from the comparison script, or a ROUGE library, for the formal evaluation (issue #5).
- Formula-heavy articles lose inline math in body sentences; consider keeping `alttext` if full-article output is ever shown.
