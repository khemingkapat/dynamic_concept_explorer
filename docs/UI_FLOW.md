# Dynamic Concept Explorer — UI Walkthrough & Screenshot Flow

A reusable end-to-end flow for demonstrating, testing, and capturing screenshots of the 2-page Streamlit application (`src/dynamic_concept_explorer/app.py`).

## Prerequisites

1. Start the Streamlit app locally:
   ```bash
   uv run streamlit run src/dynamic_concept_explorer/app.py --server.port 8501
   ```
2. Open `http://localhost:8501` in a browser (1440×900 or wide desktop viewport recommended).
3. Save all captured screenshots under `docs/images/`.

---

## Step-by-Step Flow

### Step 1: Landing Page (`Home`)
- **Action:** Open `http://localhost:8501`.
- **What to verify:**
  - Top navigation shows **Home** and **Explorer**.
  - Page displays the **Dynamic Concept Explorer** title, a single **Wikipedia article URL** text input, and the **Explore** button.
- **Screenshot:** `docs/images/ui_home.png`
- **Next Action:** Enter `https://en.wikipedia.org/wiki/Text_mining` and click **Explore**.

---

### Step 2: Explorer Overview (Default Seed View)
- **Action:** Wait for the exploration pipeline to finish and render the **Explorer** page.
- **What to verify:**
  - Top KPI strip shows 6 metrics with hover `?` tooltips (**Discovered Nodes**, **Network Edges**, **Topic Clusters**, **Graph Density**, **Avg Clustering**, **Mean C_v Coherence**).
  - Left panel (**Interactive Concept Space**) renders the Apache ECharts network with:
    - 2D spatial layout from MDS on Jensen-Shannon distance between LDA document-topic distributions.
    - Node colors by dominant LDA topic + interactive legend.
    - Node sizes scaled by PageRank centrality (gold ring on the seed node `Text mining`).
    - Dashed background grid (`silent: true`) with pan and scroll-zoom (`roam: true`).
  - Right panel (**Node Inspector**) defaults to the seed node (`Text mining`), displaying its **Algorithmic Summary** (`TextRank`, `Full article` scope, `0.50` length scale) and **ROUGE-1 / ROUGE-2 / ROUGE-L F1** cards directly below.
- **Screenshot:** `docs/images/ui_explorer_overview.png`

---

### Step 3: Interactive Node Focus (Click / Dropdown Highlighting)
- **Action:** Click a connected hub or bridge node in the graph (e.g., `Natural language processing` or `Database`), or select it from the **Selected concept** dropdown.
- **What to verify:**
  - Bidirectional sync: clicking a node updates the dropdown, and changing the dropdown highlights the node in the graph.
  - Selected node shows a white glow ring; direct neighbors and incident edges remain bright while non-neighbors dim.
  - Right panel updates immediately to the selected concept's summary, LDA topic badge + confidence, and ROUGE-1/2/L scores.
- **Screenshot:** `docs/images/ui_node_focus.png`

---

### Step 4: Node-Level Deep-Dive Tools
- **Action:** In the right-hand **Node Inspector**, expand:
  1. **Compare TextRank vs LexRank on this node** → click **Run comparison** to render the 4-variant (`TextRank/LexRank` × `Full/Lead`) latency and ROUGE table.
  2. **Section WordCloud** → display the concept's section-keyword WordCloud.
- **What to verify:**
  - Both expanders render cleanly without overlapping or clipping.
- **Screenshot:** `docs/images/ui_node_details.png`

---

### Step 5: Full Sidebar Parameter Tuning
- **Action:** In the left sidebar under **Exploration Settings**, adjust parameters across all three pipeline stages and click **Run Exploration**:
  - **1. Network & Pruning:** Set `Top-K candidate links` = `20`, `Minimum in-degree` = `2` (aligns with the `Text_mining` case study: 13 nodes, 36 edges).
  - **2. Summarizer:** Adjust `Summarization method` (`TextRank` / `LexRank`), `Article scope` (`Full article` / `Lead section`), and `Summary length scale (0-1)` (scales between 1 sentence at `0.0` and `len(lead) / len(full)` at `1.0`).
  - **3. Topic Modeling (LDA):** Set `Number of topics` = `4`, expand **LDA priors & vocabulary filter**, and tune `alpha` (`0.01–1.00`), `eta` (`0.01–1.00`), `no_below`, `no_above`, and optional `Custom stopwords`.
- **What to verify:**
  - Sidebar sliders/expanders are visible in the capture alongside the updated graph and KPI metrics.
- **Screenshot:** `docs/images/ui_sidebar_tuning.png`

---

### Step 6: Quantitative Evaluation & Pruning Breakdown
- **Action:** Scroll down below the main graph workspace to the **Quantitative Evaluation** section and expand **Per-Document ROUGE & 2-Pass Pruning Breakdown**.
- **What to verify:**
  - **Topic Coherence ($C_v$)** displays the macro-averaged $C_v$ score and per-topic $C_v$ + top keywords table.
  - **Corpus ROUGE & Graph Topology** displays Mean ROUGE-1/2/L F1 and network metrics (Node Count, Edge Count, Density, Reciprocity, Avg Clustering).
  - Expanded breakdown shows per-article ROUGE scores and pruned candidate links.
- **Screenshot:** `docs/images/ui_evaluation.png`

---

## Report Integration Checklist

1. Confirm all 6 images exist in `docs/images/`:
   - `ui_home.png`
   - `ui_explorer_overview.png`
   - `ui_node_focus.png`
   - `ui_node_details.png`
   - `ui_sidebar_tuning.png`
   - `ui_evaluation.png`
2. Reference them in `docs/FINAL_REPORT.md` under **Section 5.4 Interactive Prototype (Streamlit UI)** with relative paths `images/ui_*.png`.
