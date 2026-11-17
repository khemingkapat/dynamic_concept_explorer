"""Page 2: Graph-centralized Concept Explorer dashboard."""

from __future__ import annotations

import html
import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from dynamic_concept_explorer import evaluation, scraper, summarizer, visualization
from dynamic_concept_explorer.explorer_helpers import (
    TOPIC_COLORS,
    build_echarts_graph_spec,
    compute_target_sentences,
    get_cache_stats,
    run_exploration,
    run_single_benchmark,
)


def _extract_selected_node(chart_state: object, nodes: list[str]) -> str | None:
    """Safely extract the clicked node ID from an ECharts selection state."""
    if not chart_state or not hasattr(chart_state, "selection"):
        return None
    selection = chart_state.selection
    if not selection:
        return None
    selected_groups = (
        selection.get("selected")
        if isinstance(selection, dict)
        else getattr(selection, "selected", None)
    )
    if not selected_groups:
        return None
    for group in selected_groups:
        data_type = group.get("data_type") if isinstance(group, dict) else getattr(group, "data_type", None)
        if data_type not in (None, "node", "main"):
            continue
        indices = (
            group.get("data_indices")
            if isinstance(group, dict)
            else getattr(group, "data_indices", None)
        )
        if indices:
            idx = indices[0]
            if isinstance(idx, int) and 0 <= idx < len(nodes):
                return nodes[idx]
    return None


st.title("Dynamic Concept Explorer")

with st.sidebar:
    st.subheader("Exploration Settings")

    with st.form("explore_form"):
        wiki_url = st.text_input(
            "Wikipedia article URL",
            value=st.session_state.get("wiki_url", ""),
            placeholder="https://en.wikipedia.org/wiki/Machine_learning",
            help="Enter a full English Wikipedia article URL.",
            key="sidebar_wiki_url",
        )

        st.markdown("**1. Network & Pruning**")
        top_k = st.slider(
            "Top-K candidate links",
            min_value=5,
            max_value=25,
            value=10,
            help="Maximum number of candidate links selected in the first pass.",
            key="sidebar_top_k",
        )
        min_in_degree = st.slider(
            "Minimum in-degree",
            min_value=0,
            max_value=5,
            value=1,
            help="2-pass neighborhood in-degree threshold to prune peripheral links.",
            key="sidebar_min_in_degree",
        )

        st.markdown("**2. Summarizer**")
        method = st.selectbox(
            "Summarization method",
            options=("textrank", "lexrank"),
            format_func={
                "textrank": "TextRank (Extractive Graph)",
                "lexrank": "LexRank (Corpus Centroid)",
            }.get,
            key="sidebar_method",
        )
        scope = st.selectbox(
            "Article scope",
            options=("full", "lead"),
            index=0,
            format_func={"full": "Full article", "lead": "Lead section"}.get,
            key="sidebar_scope",
        )
        summary_scale = st.slider(
            "Summary length scale (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=(
                "Scaled between 1 sentence (0.0) and the Wikipedia lead length ratio "
                "len(lead)/len(full) (1.0)."
            ),
            key="sidebar_summary_scale",
        )

        st.markdown("**3. Topic Modeling (LDA)**")
        num_topics = st.slider(
            "Number of topics",
            min_value=2,
            max_value=6,
            value=3,
            key="sidebar_num_topics",
        )
        extra_stopwords = st.text_input(
            "Custom stopwords",
            placeholder="e.g. also, use, model, study",
            help="Comma-separated additional English stopwords.",
            key="sidebar_extra_stopwords",
        )
        with st.expander("LDA priors & vocabulary filter"):
            alpha = st.slider(
                "Document-topic prior (alpha)",
                min_value=0.01,
                max_value=1.00,
                value=0.33,
                step=0.01,
                help=(
                    "Controls how many topics each article spans. "
                    "Lower values produce sparser, more focused assignments."
                ),
                key="sidebar_lda_alpha",
            )
            eta = st.slider(
                "Topic-word prior (eta)",
                min_value=0.01,
                max_value=1.00,
                value=0.33,
                step=0.01,
                help=(
                    "Controls how many words define each topic. "
                    "Lower values produce sharper topic keywords."
                ),
                key="sidebar_lda_eta",
            )
            no_below = st.slider(
                "Min doc count (no_below)",
                min_value=1,
                max_value=5,
                value=2,
                help="Ignore words appearing in fewer than this many articles.",
                key="sidebar_lda_no_below",
            )
            no_above = st.slider(
                "Max doc fraction (no_above)",
                min_value=0.30,
                max_value=1.00,
                value=0.85,
                step=0.05,
                help="Ignore words appearing in more than this fraction of articles.",
                key="sidebar_lda_no_above",
            )

        submitted = st.form_submit_button(
            "Run Exploration",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    st.divider()
    cache_stats = get_cache_stats()
    st.caption(f"Page cache: {cache_stats['count']} files (~{cache_stats['size_kb']} KB)")
    if st.button(
        "Clear Wikipedia cache",
        icon=":material/delete:",
        width="stretch",
        key="clear_cache_btn",
    ):
        scraper.clear_page_cache()
        st.success("Wikipedia cache cleared.")
        st.rerun()

# Trigger exploration when submitted from sidebar or navigated from Home
pending = st.session_state.pop("pending_explore", False)
if submitted or pending:
    active_url = wiki_url if submitted else st.session_state.get("wiki_url", "")
    st.session_state["wiki_url"] = active_url
    st.session_state["exploration"] = None
    st.session_state.pop("selected_node", None)
    st.session_state.pop("node_dropdown_sync", None)
    st.session_state.pop("last_chart_clicked_node", None)
    try:
        with st.spinner(f"Exploring and analyzing {active_url}..."):
            st.session_state["exploration"] = run_exploration(
                active_url,
                top_k,
                min_in_degree,
                method,
                scope,
                summary_scale,
                num_topics,
                extra_stopwords,
                alpha=alpha,
                eta=eta,
                no_below=no_below,
                no_above=no_above,
            )
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Exploration failed: {exc}", icon=":material/error:")

result = st.session_state.get("exploration")
if not result:
    st.info(
        "Enter a full Wikipedia URL on the Home page or in the sidebar and click **Run Exploration**.",
        icon=":material/info:",
    )
    st.stop()

# --- EXTRACT EXPLORATION RESULTS ---
nodes: list[str] = result["nodes"]
topic_labels: dict[str, str] = result["labels"]
assignments = result["assignments"]
clustering = result["clustering"]
topology: evaluation.GraphTopologyMetrics = result["topology"]
coherence: evaluation.TopicCoherenceResult | None = result["topic_coherence"]
seed_title: str = result["seed_title"]

if "selected_node" not in st.session_state or st.session_state["selected_node"] not in nodes:
    st.session_state["selected_node"] = result["seed_id"] if result["seed_id"] in nodes else nodes[0]

st.subheader(f"Concept Map: {seed_title}")

# --- TOP KPI METRICS BAR ---
kpi_cols = st.columns(6)
kpis = [
    (
        "Discovered Nodes",
        str(topology.node_count),
        "Total Wikipedia articles in the pruned concept network (including the seed article).",
    ),
    (
        "Network Edges",
        str(topology.edge_count),
        "Number of directed Wikipedia hyperlink connections among the discovered concepts.",
    ),
    (
        "Topic Clusters",
        str(clustering.num_topics if clustering else 0),
        "Number of latent topics discovered by LDA across the concept articles.",
    ),
    (
        "Graph Density",
        f"{topology.density:.3f}",
        "Ratio of actual edges to all possible directed edges (0 = sparse, 1 = fully connected).",
    ),
    (
        "Avg Clustering",
        f"{topology.average_clustering:.3f}",
        "Average tendency of connected concepts to form tight local clusters (0 to 1).",
    ),
    (
        "Mean C_v Coherence",
        f"{coherence.mean_coherence:.3f}" if coherence else "N/A",
        "Gensim C_v topic coherence averaged across topics (> 0.4 generally indicates coherent keywords).",
    ),
]
for col, (label, value, tooltip) in zip(kpi_cols, kpis, strict=True):
    with col, st.container(border=True):
        st.metric(label=label, value=value, help=tooltip)

# =============================================================================
# CENTRAL WORKSPACE: INTERACTIVE GRAPH (LEFT) + NODE INSPECTOR (RIGHT)
# =============================================================================
col_graph, col_inspector = st.columns([1.4, 1], gap="large")

with col_graph:
    st.markdown("### Interactive Concept Space")
    st.caption(
        "**Distance:** Closer nodes share more similar LDA topic mixtures (MDS on Jensen-Shannon distance).  \n"
        "**Size & Color:** Bubble size reflects PageRank centrality in the Wikipedia network; "
        "color shows the dominant LDA topic (gold ring = seed article).  \n"
        "**Interaction:** Click any node or use the dropdown to highlight its connections and inspect its summary."
    )

    echarts_spec = build_echarts_graph_spec(
        result["seed_id"],
        nodes,
        result["edges"],
        topic_labels,
        assignments,
        clustering,
        pagerank_scores=topology.pagerank,
        selected_node=st.session_state["selected_node"],
    )
    with st.container(border=True):
        chart_state = st.echarts_chart(
            echarts_spec,
            height=540,
            width="stretch",
            key="concept_graph_chart",
            on_select="rerun",
            alt="Interactive concept network positioned by document-topic similarity",
        )

    clicked_node = _extract_selected_node(chart_state, nodes)
    if clicked_node and clicked_node != st.session_state.get("last_chart_clicked_node"):
        st.session_state["last_chart_clicked_node"] = clicked_node
        st.session_state["selected_node"] = clicked_node
        st.session_state["node_dropdown_sync"] = clicked_node
        st.rerun()

    if result["clustering_warning"]:
        st.warning(f"Clustering note: {result['clustering_warning']}")

with col_inspector:
    # Synced node dropdown (reflects graph clicks and allows direct keyboard selection)
    def _on_select_change() -> None:
        st.session_state["selected_node"] = st.session_state["node_dropdown_sync"]

    current_idx = nodes.index(st.session_state["selected_node"])
    st.selectbox(
        "Selected concept",
        options=nodes,
        index=current_idx,
        format_func=lambda n: f"{topic_labels[n]} ({'Seed' if n == result['seed_id'] else 'Concept'})",
        key="node_dropdown_sync",
        on_change=_on_select_change,
    )

    selected_node = st.session_state["selected_node"]
    article = result["articles"][selected_node]
    summary = result["summaries"][selected_node]
    node_rouge_map = result.get("node_rouge") or {}
    if selected_node in node_rouge_map:
        rouge_doc = node_rouge_map[selected_node]
    elif article.lead.strip() and summary.text.strip():
        rouge_doc = evaluation.compute_rouge_score(summary.text, article.lead)
    else:
        rouge_doc = None
    assignment = assignments.get(selected_node)
    node_sections = result["section_keywords"].get(selected_node, {})

    header_col, link_col = st.columns([2, 1], vertical_alignment="center")
    with header_col:
        st.markdown(f"### {topic_labels[selected_node]}")
    with link_col:
        st.link_button(
            "Wikipedia",
            f"https://en.wikipedia.org{selected_node}",
            icon=":material/open_in_new:",
            width="stretch",
        )

    if assignment and clustering:
        desc = clustering.topics[assignment.topic_id]
        color = TOPIC_COLORS[assignment.topic_id % len(TOPIC_COLORS)]
        st.markdown(
            f'<span style="display:inline-block;border-radius:20px;padding:4px 12px;'
            f"font-size:0.85rem;font-weight:600;background:{color};color:#12101e\">"
            f"Topic {assignment.topic_id + 1} · {html.escape(desc.label)}</span> "
            f'<span style="font-size:0.85rem;opacity:0.8">Confidence: {assignment.confidence * 100:.1f}%</span>',
            unsafe_allow_html=True,
        )

    st.markdown("**Algorithmic Summary**")
    with st.container(border=True):
        if summary.text:
            st.write(summary.text)
        else:
            st.warning("Article text is too short or no sentences met the length criteria.")

        lead_ratio = (len(article.lead) / len(article.full)) if article.full else 0.0
        st.caption(
            f"{summary.method.title()} · Scope: {summary.scope.title()} · "
            f"{len(summary.sentences)} sentence(s) · Max lead ratio: {lead_ratio:.2f} · "
            f"{summary.elapsed_ms:.1f} ms"
        )

    # ROUGE-1, ROUGE-2, ROUGE-L right below the summary
    r1_col, r2_col, rl_col = st.columns(3)
    with r1_col, st.container(border=True):
        st.metric(
            "ROUGE-1 F1",
            f"{rouge_doc.rouge1.fmeasure:.3f}" if rouge_doc else "N/A",
            help="Unigram (single-word) overlap F1 against the Wikipedia lead abstract.",
        )
    with r2_col, st.container(border=True):
        st.metric(
            "ROUGE-2 F1",
            f"{rouge_doc.rouge2.fmeasure:.3f}" if rouge_doc else "N/A",
            help="Bigram (two-word phrase) overlap F1 against the Wikipedia lead abstract.",
        )
    with rl_col, st.container(border=True):
        st.metric(
            "ROUGE-L F1",
            f"{rouge_doc.rougeL.fmeasure:.3f}" if rouge_doc else "N/A",
            help="Longest common subsequence F1 against the Wikipedia lead abstract.",
        )

    # Node-level details in clean expanders
    with st.expander("Compare TextRank vs LexRank on this node"):
        if st.button("Run comparison", key=f"bench_btn_{selected_node}", width="stretch"):
            bench_data = run_single_benchmark(article.lead, article.body, summary_scale)
            df_bench = pd.DataFrame(
                [
                    {
                        "Method": item["label"],
                        "ms": round(item["elapsed_ms"], 1),
                        "R-1": round(item["rouge1_f1"], 3),
                        "R-2": round(item["rouge2_f1"], 3),
                        "R-L": round(item["rougeL_f1"], 3),
                    }
                    for item in bench_data
                ]
            )
            st.dataframe(df_bench, hide_index=True, width="stretch")

    with st.expander("Section WordCloud"):
        all_keywords = [kw for kws in node_sections.values() for kw in kws]
        if all_keywords:
            wc = visualization.generate_section_wordcloud(
                all_keywords,
                section_title=f"Keywords: {topic_labels[selected_node]}",
                background_color="#171622",
                show=False,
            )
            if wc:
                st.image(
                    wc.to_image(),
                    caption=f"WordCloud — {topic_labels[selected_node]}",
                    width="stretch",
                )
                plt.close("all")
            else:
                st.warning("Not enough keywords to generate a WordCloud.")
        else:
            st.info("No section keywords extracted for this article.")

    with st.expander("Wikipedia Lead Abstract & Full Text"):
        st.markdown("**Lead Abstract**")
        st.write(article.lead if article.lead else "(No lead text available)")
        st.divider()
        st.markdown("**Full Article Text**")
        st.text_area(
            "Full prose content",
            value=article.full,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    if assignment and clustering:
        with st.expander("LDA Topic Distribution & Section Keywords"):
            for tid, prob in sorted(assignment.distribution, key=lambda x: x[0]):
                t_label = clustering.topics[tid].label
                st.caption(f"Topic {tid + 1} ({t_label}): {prob * 100:.1f}%")
                st.progress(float(prob))
            if node_sections:
                st.divider()
                for sec_title, kw_list in list(node_sections.items())[:6]:
                    clean_title = re.sub(r"\[edit\]", "", sec_title).strip()
                    if kw_list:
                        st.markdown(f"**{clean_title}**: {', '.join(kw_list[:15])}")

    # Re-summarize interactively on this node without re-scraping
    with st.expander("Tune summary length & method for this node"):
        t_col1, t_col2 = st.columns(2)
        tune_method = t_col1.selectbox(
            "Method", ("textrank", "lexrank"), index=0, key="tune_method"
        )
        tune_scope = t_col2.selectbox("Scope", ("full", "lead"), index=0, key="tune_scope")
        tune_scale = st.slider(
            "Length scale (0-1)", 0.0, 1.0, float(summary_scale), 0.05, key="tune_scale"
        )
        n_sents, eff_ratio = compute_target_sentences(article.lead, article.full, tune_scale)
        tuned = summarizer.summarize_node(
            article.lead, article.body, method=tune_method, scope=tune_scope, n_sentences=n_sents
        )
        tuned_rouge = (
            evaluation.compute_rouge_score(tuned.text, article.lead)
            if article.lead.strip() and tuned.text.strip()
            else None
        )
        st.write(tuned.text)
        st.caption(
            f"{len(tuned.sentences)} sentence(s) (ratio {eff_ratio:.2f}) · "
            + (
                f"R-1: {tuned_rouge.rouge1.fmeasure:.3f} · "
                f"R-2: {tuned_rouge.rouge2.fmeasure:.3f} · "
                f"R-L: {tuned_rouge.rougeL.fmeasure:.3f}"
                if tuned_rouge
                else "ROUGE N/A"
            )
        )

# =============================================================================
# BELOW THE GRAPH: QUANTITATIVE EVALUATION (C_v, TOPOLOGY, CORPUS ROUGE)
# =============================================================================
st.divider()
st.markdown("### Quantitative Evaluation")

eval_col1, eval_col2 = st.columns(2, gap="large")

with eval_col1:
    st.markdown("#### Topic Coherence ($C_v$)")
    if coherence and coherence.num_topics > 0:
        st.metric("Macro-Averaged C_v Coherence", f"{coherence.mean_coherence:.4f}")
        st.caption("Reference: C_v > 0.4 generally indicates coherent topic keywords.")

        coh_rows = []
        for tid, c_score in sorted(coherence.topic_coherences.items()):
            t_desc = clustering.topics.get(tid)
            label = t_desc.label if t_desc else f"Topic {tid}"
            coh_rows.append(
                {
                    "Topic": f"Topic {tid + 1}",
                    "C_v Score": round(c_score, 4),
                    "Top Keywords": label,
                }
            )
        st.dataframe(pd.DataFrame(coh_rows), hide_index=True, width="stretch")
    else:
        st.info("Topic coherence unavailable (requires at least 2 articles with sufficient vocabulary).")

with eval_col2:
    st.markdown("#### Corpus ROUGE & Graph Topology")
    rouge_eval = result.get("rouge_eval")
    if rouge_eval:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean ROUGE-1 F1", f"{rouge_eval.mean_rouge1.fmeasure:.4f}")
        c2.metric("Mean ROUGE-2 F1", f"{rouge_eval.mean_rouge2.fmeasure:.4f}")
        c3.metric("Mean ROUGE-L F1", f"{rouge_eval.mean_rougeL.fmeasure:.4f}")
    else:
        st.info("Corpus ROUGE evaluation unavailable.")

    topo_rows = [
        {"Metric": "Node Count", "Value": str(topology.node_count)},
        {"Metric": "Edge Count", "Value": str(topology.edge_count)},
        {"Metric": "Density", "Value": f"{topology.density:.4f}"},
        {"Metric": "Reciprocity", "Value": f"{topology.reciprocity:.4f}"},
        {"Metric": "Avg Clustering", "Value": f"{topology.average_clustering:.4f}"},
    ]
    st.dataframe(pd.DataFrame(topo_rows), hide_index=True, width="stretch")

with st.expander("Per-Document ROUGE & 2-Pass Pruning Breakdown"):
    if rouge_eval:
        per_doc_rows = [
            {
                "Article": topic_labels.get(doc_id, doc_id),
                "ROUGE-1 F1": round(res.rouge1.fmeasure, 3),
                "ROUGE-2 F1": round(res.rouge2.fmeasure, 3),
                "ROUGE-L F1": round(res.rougeL.fmeasure, 3),
            }
            for doc_id, res in rouge_eval.per_document.items()
        ]
        st.dataframe(pd.DataFrame(per_doc_rows), hide_index=True, width="stretch")

    total_candidates = len(result["neighborhood"].candidates) + len(result["neighborhood"].pruned)
    st.caption(
        f"2-Pass Pruning: {len(result['neighborhood'].candidates)} accepted / "
        f"{total_candidates} total candidates (min in-degree >= {min_in_degree})."
    )
    if result["neighborhood"].pruned:
        pruned_rows = [
            {"Pruned Concept": p.text, "In-Degree": p.in_degree, "Initial Score": round(p.score, 3)}
            for p in result["neighborhood"].pruned
        ]
        st.dataframe(pd.DataFrame(pruned_rows), hide_index=True, width="stretch")
