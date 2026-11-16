# Dynamic Concept Explorer: Applying Text Analytics to Everyday Knowledge Discovery

---

## 1. Introduction

### 1.1 Background and Intent

Most text analytics projects begin with a static dataset—such as a pre-collected table of product reviews or social media posts—and apply a fixed sequence of models to produce a one-time report. **Dynamic Concept Explorer** takes a different starting point. Rather than analyzing a single fixed corpus, this project is an **experimental workflow and prototype tool** built to answer a practical, everyday question:

> *"How can the text analytics techniques learned in class actually help us in a day-to-day task?"*

A common day-to-day task for students and researchers is **learning an unfamiliar technical or academic subject on Wikipedia**. In practice, this often leads to cognitive overload: a single article contains dozens of hyperlinked sub-concepts, and following those links across tabs quickly causes readers to lose track of how ideas relate to one another or spend excessive time reading long articles just to grasp basic definitions.

To test whether text analytics can solve this problem, we built a modular pipeline that dynamically ingests a seed Wikipedia topic, prunes its surrounding hyperlink neighborhood into a manageable concept graph, clusters related concepts into themes, generates short node overviews, and evaluates the results quantitatively. Throughout the project, our focus is **not** simply on running every algorithm, but on **evaluating which techniques genuinely fit an interactive, on-demand workflow—and which do not**.

### 1.2 Objectives

1. **Dynamic Concept Ingestion:** Construct an on-demand concept neighborhood from any user-provided Wikipedia seed URL, filtering out boilerplate and peripheral links to prevent graph explosion.
2. **Automated Content Distillation:** Test extractive (TextRank, LexRank) and abstractive (BART) summarization methods to determine which can produce fast, definitional node overviews suitable for interactive exploration.
3. **Unsupervised Conceptual Organization:** Apply Latent Dirichlet Allocation (LDA) to group discovered concept nodes into interpretable thematic clusters without manual labels.
4. **Quantitative & Practical Evaluation:** Measure graph topology, topic coherence ($C_v$), summarization ROUGE scores, and runtime latency to assess how well each course technique translates to a day-to-day learning tool.

---

## 2. Dynamic Data Acquisition & Preparation

Unlike traditional studies that scrape a large static dataset upfront, our "dataset" is **constructed dynamically per seed topic** through a two-pass scraping and preprocessing pipeline ([`scraper.py`](../src/dynamic_concept_explorer/scraper.py)).

### 2.1 Seed-Driven Data Acquisition (2-Pass Discovery)

Given a seed Wikipedia URL (e.g., `https://en.wikipedia.org/wiki/Text_mining`), the pipeline parses `div#bodyContent` using `BeautifulSoup` and executes two passes:

1. **Pass 1 — Outbound Candidate Ranking:**
   - Extracts internal Wikipedia links (`/wiki/...`), normalizes absolute/relative URLs, strips URL fragments (`#...`) and query parameters, and removes administrative namespaces (`Special:`, `Help:`, `File:`, `Category:`, etc.).
   - Skips non-content boilerplate sections (`See also`, `References`, `External links`, `Notes`, `Further reading`, `Bibliography`, `Sources`).
   - Ranks outbound links using a lightweight heuristic that prioritizes concepts introduced in the article's opening lead section and mentioned repeatedly:

   $$\text{score}(v) = 2.0 \cdot \mathbb{I}[\text{in\_lead}(v)] + \ln(1 + \text{count}(v))$$

   - Retains the top $k = 20$ candidate links.

2. **Pass 2 — Neighborhood In-Degree Pruning:**
   - Fetches each of the 20 candidate pages (with a two-tier in-memory and 7-day JSON disk cache in `.cache/wiki/` to support fast, repeatable experimentation).
   - Counts how many other top-$k$ candidates link *to* each candidate (its **neighborhood in-degree**).
   - Prunes weakly connected peripheral nodes with $\text{in\_degree} < 2$ (with a `min_nodes = 5` fallback so sparse topics never collapse).

### 2.2 Article Text Extraction & Preprocessing

For every active node (seed + kept candidates), the pipeline extracts clean prose into a structured `ArticleText(lead, body, full)` object:

* **Lead vs. Body Split:** Splits content at the first `<h2>` heading so the opening abstract (`lead`) and the rest of the article (`body`) can be analyzed separately or together (`full`).
* **Noise & Boilerplate Removal:** Strips Wikipedia citation markers (`[1]`, `[citation needed]`), inline math markup, edit links, and infobox/table text while skipping boilerplate sections.
* **NLP Preprocessing (for Topic Modeling, [`topic_modeling.py`](../src/dynamic_concept_explorer/topic_modeling.py)):**
  1. Tokenizes text with `gensim.utils.simple_preprocess` (lowercasing, de-accenting, punctuation removal, minimum token length $\ge 3$).
  2. Applies two-pass WordNet lemmatization (verb pass followed by noun pass) to normalize inflected forms.
  3. Removes English stopwords (plus optional seed-specific stopwords).
  4. Filters extreme vocabulary terms (`no_below=2`, `no_above=0.85`), falling back to `no_below=1, no_above=1.0` if strict filtering empties small neighborhoods.

---

## 3. Methodology & Technique Selection ("What Fits & Why")

A central goal of this project is to test how standard text analytics methods behave under the constraints of an **interactive, day-to-day tool** (low latency, small dynamic corpora of ~10–25 documents, and noisy web-scraped text).

### 3.1 Graph Construction & Topology Analysis

To understand the structure of the pruned concept network, we model the neighborhood as a directed graph $G = (V, E)$ and compute structural metrics ([`evaluation.py`](../src/dynamic_concept_explorer/evaluation.py)):

* **Density & Reciprocity:** Measure how tightly interconnected the discovered concepts are and how often concepts mutually reference one another.
* **Average Clustering Coefficient (Undirected View):** Measures local triadic closure—whether neighbors of a concept also link to each other.
* **Centrality & PageRank:** Identify foundational "hub" concepts (high in-degree / PageRank) and conceptual "bridges" (high betweenness centrality) within the domain.

### 3.2 Text Distillation / Summarization

We implemented and compared all three summarization approaches covered in our workflow ([`summarizer.py`](../src/dynamic_concept_explorer/summarizer.py)), targeting a concise **2-sentence overview** per node:

1. **Extractive TextRank (Default):**
   - Splits text into sentences (NLTK `sent_tokenize`), filters out fragments outside 6–60 words (capped at 250 sentences), builds L2-normalized TF-IDF vectors, computes the zero-diagonal cosine similarity matrix $\mathbf{S}$, and runs `networkx.pagerank` ($d = 0.85$) to select the top 2 sentences in original chronological order.
   - **Scope Comparison (`lead` vs. `full`):** We evaluate TextRank both on the full article (`scope="full"`) and on the lead abstract (`scope="lead"`, with automatic fallback to `full` if the lead has fewer than 2 usable sentences).
2. **Extractive LexRank:**
   - Uses the `lexrank` package with a cosine threshold of $0.25$ to identify central sentences via eigenvector centrality.
3. **Abstractive Neural Seq2Seq (BART):**
   - Uses `facebook/bart-large-cnn` (1024-token input limit, output trimmed to 2 complete sentences, with automatic fallback to TextRank if unavailable).
   - Tested on-demand rather than as a default to evaluate whether neural abstractive fluency justifies CPU inference cost in a multi-node graph.

### 3.3 Unsupervised Topic Modeling (LDA)

To organize the concept graph into thematic groups without predefined labels, we apply **Latent Dirichlet Allocation (LDA)** (`gensim.models.LdaModel`, `num_topics=4`, `passes=10`, `alpha="symmetric"`) over the active node documents. Each node is assigned its dominant topic and confidence score, and topic quality is evaluated quantitatively using **Gensim $C_v$ topic coherence** (`top_n=10`).

### 3.4 Quantitative Evaluation Metrics

Because our system has no external human-written gold labels for arbitrary Wikipedia topics, we evaluate outputs using intrinsic and reference-proxy metrics:
* **Graph Topology:** Node/edge count, density, average clustering, reciprocity, PageRank, in-degree centrality, and betweenness centrality.
* **Topic Quality:** Per-topic and mean Gensim $C_v$ coherence.
* **Summarization Proxy:** Macro-averaged **ROUGE-1, ROUGE-2, and ROUGE-L** (precision, recall, F1) comparing full-article summaries (`TextRank-full`, `LexRank-full`) against each Wikipedia article's human-written lead abstract as a reference. (Note: `TextRank-lead` selects directly from the lead, so scoring it against the lead would be circular; it is evaluated via latency and qualitative inspection.)

---

## 4. Preliminary Results & Discussion (Case Study: `Text_mining`)

To demonstrate the pipeline on a course-relevant domain, we ran the full workflow on the seed article **`https://en.wikipedia.org/wiki/Text_mining`**.

### 4.1 Neighborhood Discovery & Corpus Characteristics

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Pass-1 Outbound Candidates** | 20 | Ranked by $2.0 \cdot \text{in\_lead} + \ln(1 + \text{count})$ |
| **Kept Candidates (Pass 2)** | 12 | `in_degree >= 2` across the top-20 neighborhood |
| **Pruned Candidates** | 8 | `in_degree < 2` |
| **Active Graph Size** | 13 nodes, 36 edges | Seed (`Text_mining`) + 12 kept neighbors |
| **Total Corpus Volume** | 30,836 words | Across 13 fetched Wikipedia articles |
| **Full Article Length (words)** | Min: 841 / Median: 2,296 / Max: 6,667 | Mean: 2,372.0 words |
| **Lead Section Length (words)** | Min: 62 / Median: 221 / Max: 604 | Mean: 260.9 words |

**Top Kept vs. Pruned Nodes (Selection):**

* **Top Kept Neighbors:** `Natural_language_processing` (`in_degree=8`, `score=3.099`), `Data_mining` (`in_degree=7`, `score=3.386`), `Information_retrieval` (`in_degree=6`, `score=3.099`), `Document` (`4`), `Information_extraction` (`4`), `Database` (`3`), `Information` (`3`), `Named_entity_recognition` (`2`), `Parsing` (`2`), and `Predictive_analytics` (`2`).
* **Pruned Neighbors:** Generic/peripheral links such as `Book` (`1`), `Email` (`1`), and `Annotation` (`1`) were successfully filtered out. However, niche subfields like `Sentiment_analysis` (`in_degree=1`) and `Document_summarization` (`in_degree=0`) were also pruned because few of the other top-20 articles linked back to them.

**Observation:** 2-pass in-degree pruning effectively prevents graph explosion and removes generic noise (`Book`, `Email`), but it introduces a **precision–recall trade-off**: specialized subtopics that are only linked from the seed page can be dropped if `min_in_degree=2` is strictly enforced.

### 4.2 Graph Topology Results

For the 13-node, 36-edge `Text_mining` graph:

* **Density:** `0.2308`
* **Average Clustering Coefficient:** `0.6296` (high local triadic closure among core concepts)
* **Reciprocity:** `0.4444` (nearly half of all concept links are bidirectional)

| Node | PageRank | In-Degree Centrality | Betweenness Centrality | Role in Neighborhood |
| :--- | :---: | :---: | :---: | :--- |
| `/wiki/Database` | **0.1608** | 0.1667 | **0.3598** | Structural bridge |
| `/wiki/Natural_language_processing` | 0.1498 | **0.5000** | 0.2727 | Core hub |
| `/wiki/Information_retrieval` | 0.1441 | 0.4167 | 0.3030 | Core hub & bridge |
| `/wiki/Data_mining` | 0.1095 | 0.3333 | 0.2348 | Core hub |
| `/wiki/Information` | 0.0776 | 0.2500 | 0.1402 | Supporting concept |

**Observation:** Graph centrality metrics immediately distinguish **foundational hubs** (`Natural_language_processing`, `Information_retrieval`, `Data_mining`) from **structural bridges** (`Database`), which could be used to guide a learner's reading order.

### 4.3 Unsupervised LDA Topic Modeling Results

After preprocessing and extreme-term filtering (`no_below=2`, `no_above=0.85`), the vocabulary contained **1,344 terms** across the 13 documents. Training a 4-topic LDA model yielded a **mean $C_v$ coherence of 0.4619**:

| Topic | $C_v$ Coherence | Top Keywords | Assigned Nodes (Count) |
| :--- | :---: | :--- | :--- |
| **Topic 0** | 0.4230 | `mine`, `text`, `document`, `pattern`, `analysis` | `Text_mining`, `Natural_language_processing`, `Data_mining`, `Document`, `Information_extraction`, `Information` (6) |
| **Topic 1** | 0.5455 | `relationship`, `entity`, `database`, `type`, `represent` | `Entity–relationship_model` (1) |
| **Topic 2** | 0.3974 | `algorithm`, `parse`, `grammar`, `sentence`, `entity` | `Named_entity_recognition`, `Parsing`, `Algorithm` (3) |
| **Topic 3** | 0.4817 | `database`, `application`, `query`, `relational`, `use` | `Information_retrieval`, `Database`, `Predictive_analytics` (3) |

**Observation:** Even on a small 13-document corpus, LDA separated **linguistic/parsing concepts** (Topic 2) from **database/retrieval systems** (Topic 3) and **core text/data mining** (Topic 0). At the same time, Topic 1 illustrates a practical limitation of LDA on small dynamic graphs: a specialized article (`Entity–relationship_model`) can become a singleton cluster.

### 4.4 Summarization Benchmark (`Text_mining` Neighborhood + Multi-Topic Findings)

#### A. Extractive Comparison on the 13-Node `Text_mining` Graph

| Method & Scope | Mean Latency (ms) | Median Latency (ms) | ROUGE-1 F1 (vs. Lead) | ROUGE-2 F1 (vs. Lead) | ROUGE-L F1 (vs. Lead) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TextRank (`lead`)** | **9.5** | **4.9** | — *(circular)* | — *(circular)* | — *(circular)* |
| **TextRank (`full`)** | 21.9 | 16.1 | **0.2956** | **0.2117** | **0.2591** |
| **LexRank (`full`)** | 259.9 | 238.4 | 0.1363 | 0.0143 | 0.0887 |

**Sample Outputs for `/wiki/Text_mining`:**
* **TextRank (`lead`, 59.5 ms):** *"Text mining, text data mining (TDM) or text analytics is the process of deriving high-quality information from text. (2005), there are three perspectives of text mining: information extraction, data mining, and knowledge discovery in databases (KDD)."*
* **TextRank (`full`, 25.3 ms):** *"Text mining, text data mining (TDM) or text analytics is the process of deriving high-quality information from text. Text analytics describes a set of linguistic, statistical, and machine learning techniques that model and structure the information content of textual sources for business intelligence, exploratory data analysis, research, or investigation."*
* **LexRank (`full`, 262.1 ms):** *"Text mining is also used in some email spam filters as a way of determining the characteristics of messages that are likely to be advertisements or other unwanted material. Text mining plays an important role in determining financial market sentiment."*

**Key Takeaways:**
1. **TextRank outperforms LexRank for concept definitions:** `TextRank (full)` achieved much higher ROUGE-2 (`0.2117` vs. `0.0143`) and was over **10× faster** (`21.9 ms` vs. `259.9 ms` mean). LexRank frequently selected mid-article application examples (e.g., spam filters) rather than core definitions.
2. **TextRank (`lead`) is ideal for speed, with one caveat:** Running TextRank on the lead section takes ~5–10 ms and reliably captures the opening definition, though citation-stripping can occasionally leave orphaned parentheticals such as `(2005), there are...`.

#### B. Neural Abstractive Summarization (BART) Trade-Offs

In our broader 6-topic evaluation logged during development ([`INCREMENTAL_LOG.md`](INCREMENTAL_LOG.md) — covering *Machine learning*, *Photosynthesis*, *French Revolution*, *General relativity*, *Game theory*, and *Natural language processing*), `facebook/bart-large-cnn` on CPU required **3.4–14.1 seconds per node**, compared to **3–20 ms** for `TextRank (lead)`. Furthermore, in 2 of the 6 topics (*French Revolution* and *General relativity*), BART skipped the defining opening sentence in favor of a secondary detail. Consequently, BART is **unsuitable as the default** for a 13–21 node interactive graph and is kept strictly on demand.

---

## 5. Conclusion, Limitations & Future Work

### 5.1 Conclusion: Which Text Analytics Techniques Fit Day-to-Day Use?

Returning to our core question—*how can text analytics help in a day-to-day learning task?*—our experiments show that **lightweight, structure-aware methods beat heavy models** in an interactive setting:

* **2-Pass In-Degree Pruning + PageRank/Centrality** effectively turns noisy Wikipedia hyperlinks into a compact, navigable concept map (reducing 20+ links to 12 core concepts in `Text_mining`) and highlights hub concepts automatically.
* **Extractive TextRank** (especially over the lead section) provides fast (~5–20 ms), reliable definitional summaries that outperform both LexRank (which drifts to mid-article details and is ~10× slower) and CPU-based BART (which takes several seconds per node).
* **Unsupervised LDA** provides useful high-level thematic grouping ($C_v = 0.4619$ on `Text_mining`), even on small dynamic neighborhoods, though topic counts must be kept low to avoid singleton clusters.

### 5.2 Current Limitations

1. **Formula & Citation Artifacts:** Stripping Wikipedia `<math>` elements can leave gaps in formula-heavy articles (e.g., *Game theory*), and removing `[1]` brackets can leave orphaned year parentheticals such as `(2005), ...`.
2. **Pruning Recall Trade-Off:** Requiring `in_degree >= 2` filters out generic noise (`Book`, `Email`), but can also drop legitimate niche subfields (`Sentiment_analysis`, `Document_summarization`) if they lack cross-links among the top-20 neighbors.
3. **Small-Corpus LDA Instability:** With only 10–20 documents per seed neighborhood, LDA topics are sensitive to `num_topics` and can produce singleton clusters.

### 5.3 Future Work

1. **Interactive Explorer UI (In Progress):** Wire the core pipeline (`discover_neighborhood`, `fetch_neighborhood_documents`, `summarize_node`, and `cluster_topic_nodes`) into an interactive web interface where learners can inspect node summaries, filter by LDA topic clusters, and dynamically re-center the graph.
2. **Cleaner Math & Parenthetical Handling:** Preserve `alttext` for inline Wikipedia math and strip orphaned author-year citation fragments during preprocessing.
3. **Adaptive Pruning & Topic Count Selection:** Tune `num_topics` dynamically (e.g., via $C_v$ coherence search) and allow optional 1-hop retention for high-scoring Pass-1 links.

---

## 6. References

1. Mihalcea, R., & Tarau, P. (2004). *TextRank: Bringing Order into Text*. Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing (EMNLP), 404–411.
2. Erkan, G., & Radev, D. R. (2004). *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization*. Journal of Artificial Intelligence Research, 22, 457–479.
3. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 7871–7880.
4. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993–1022.
5. Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the Space of Topic Coherence Measures*. Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM), 399–408.
6. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. Text Summarization Branches Out (ACL Workshop), 74–81.
7. Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). *Exploring Network Structure, Dynamics, and Function using NetworkX*. Proceedings of the 7th Python in Science Conference (SciPy), 11–15.
8. Řehůřek, R., & Sojka, P. (2010). *Software Framework for Topic Modelling with Large Corpora*. Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, 45–50.
