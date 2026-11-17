"""Shared pipeline and visualization helpers for the Dynamic Concept Explorer UI."""

from __future__ import annotations

import time
from urllib.parse import unquote, urlparse

import networkx as nx
import numpy as np
import requests
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

from dynamic_concept_explorer import evaluation, scraper, summarizer, topic_modeling

TOPIC_COLORS = [
    "#9b8cff",  # Soft Violet
    "#55d6be",  # Aqua Teal
    "#ffb454",  # Amber Gold
    "#ff7c91",  # Coral Rose
    "#63b3ed",  # Sky Blue
    "#b8d46a",  # Lime Sage
    "#f687b3",  # Pink
    "#4fd1c5",  # Mint
]


def validate_wikipedia_url(value: str) -> tuple[str, str]:
    """Validate a full Wikipedia article URL and return its canonical URL and title."""
    value = value.strip()
    if not value:
        raise ValueError("Please enter a Wikipedia article URL.")

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    is_wikipedia_host = hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")
    if parsed.scheme not in ("http", "https") or not is_wikipedia_host:
        raise ValueError(
            "Please enter a full Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Machine_learning)."
        )

    if not parsed.path.startswith("/wiki/") or len(parsed.path.removeprefix("/wiki/").strip("/")) == 0:
        raise ValueError(
            "URL must point to a Wikipedia article (e.g., https://en.wikipedia.org/wiki/Machine_learning)."
        )

    canonical_url = f"https://{hostname}{parsed.path}"
    slug = parsed.path.removeprefix("/wiki/")
    title = unquote(slug).replace("_", " ")
    return canonical_url, title


def compute_target_sentences(lead: str, full: str, scale: float) -> tuple[int, float]:
    """Map a 0-1 scale to a sentence count bounded by len(lead) / len(full)."""
    full_sents = summarizer.split_sentences(full)
    lead_sents = summarizer.split_sentences(lead)
    if not full_sents:
        return 1, 0.0

    max_ratio = (len(lead) / len(full)) if lead.strip() and len(full) > 0 else 0.28
    max_ratio = min(1.0, max(0.0, max_ratio))
    effective_ratio = max(0.0, min(1.0, scale)) * max_ratio
    target = max(1, round(effective_ratio * len(full_sents)))
    if lead_sents:
        target = min(target, max(1, len(lead_sents)))
    return target, effective_ratio


def compute_topic_spatial_layout(
    nodes: list[str],
    edges: dict[str, set[str]],
    assignments: dict[str, topic_modeling.NodeTopicAssignment],
) -> dict[str, tuple[float, float]]:
    """Project nodes into 2D space based on document-topic similarity (MDS on JSD)."""
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}

    if assignments and all(n in assignments for n in nodes):
        dists_list = [
            [prob for _, prob in sorted(assignments[n].distribution, key=lambda item: item[0])]
            for n in nodes
        ]
        n_nodes = len(nodes)
        jsd_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            p_i = np.asarray(dists_list[i], dtype=float)
            for j in range(i + 1, n_nodes):
                p_j = np.asarray(dists_list[j], dtype=float)
                d = float(jensenshannon(p_i, p_j))
                if np.isnan(d):
                    d = 0.0
                jsd_matrix[i, j] = d
                jsd_matrix[j, i] = d

        if np.any(jsd_matrix > 1e-6):
            mds = MDS(
                n_components=2,
                metric="precomputed",
                init="random",
                n_init=4,
                random_state=42,
            )
            coords = mds.fit_transform(jsd_matrix)
            return {
                node: (float(coords[idx, 0]) * 300.0, float(coords[idx, 1]) * 300.0)
                for idx, node in enumerate(nodes)
            }

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for src, targets in edges.items():
        if src in graph:
            for tgt in targets:
                if tgt in graph and src != tgt:
                    graph.add_edge(src, tgt)
    pos = nx.spring_layout(graph, seed=42)
    return {node: (float(x) * 300.0, float(y) * 300.0) for node, (x, y) in pos.items()}


def build_echarts_graph_spec(
    seed_id: str,
    nodes: list[str],
    edges: dict[str, set[str]],
    labels: dict[str, str],
    assignments: dict[str, topic_modeling.NodeTopicAssignment],
    clustering: topic_modeling.TopicClusteringResult | None,
    pagerank_scores: dict[str, float] | None = None,
    selected_node: str | None = None,
) -> dict:
    """Build an interactive Apache ECharts graph option colored by LDA topic and sized by PageRank."""
    coords = compute_topic_spatial_layout(nodes, edges, assignments)
    max_pr = max((pagerank_scores or {}).values(), default=1.0) or 1.0

    # Determine direct neighbors of the selected node for bidirectional highlighting
    selected_neighbors: set[str] = set()
    if selected_node and selected_node in nodes:
        selected_neighbors.add(selected_node)
        selected_neighbors.update(edges.get(selected_node, set()))
        for src, targets in edges.items():
            if selected_node in targets:
                selected_neighbors.add(src)

    categories: list[dict] = []
    if clustering:
        for tid in sorted(clustering.topics):
            desc = clustering.topics[tid]
            categories.append(
                {
                    "name": f"Topic {tid + 1}: {desc.label}",
                    "itemStyle": {"color": TOPIC_COLORS[tid % len(TOPIC_COLORS)]},
                }
            )
    else:
        categories.append({"name": "Concepts", "itemStyle": {"color": TOPIC_COLORS[0]}})

    node_items: list[dict] = []
    node_set = set(nodes)
    for node_id in nodes:
        x, y = coords.get(node_id, (0.0, 0.0))
        assignment = assignments.get(node_id)
        cat_idx = assignment.topic_id if (assignment and clustering) else 0
        pr = (pagerank_scores or {}).get(node_id, 0.0)
        size = 22.0 + 36.0 * (pr / max_pr) if pagerank_scores else 30.0
        is_seed = node_id == seed_id
        is_selected = node_id == selected_node
        is_neighbor = node_id in selected_neighbors

        item_style: dict = {}
        if is_selected:
            item_style = {
                "borderColor": "#ffffff",
                "borderWidth": 4,
                "shadowBlur": 16,
                "shadowColor": "#ffffff",
                "opacity": 1.0,
            }
        elif is_seed:
            item_style = {
                "borderColor": "#ffd166",
                "borderWidth": 3.5,
                "opacity": 1.0 if (not selected_node or is_neighbor) else 0.35,
            }
        elif selected_node:
            item_style = {"opacity": 1.0 if is_neighbor else 0.28}

        item: dict = {
            "id": node_id,
            "name": labels.get(node_id, node_id),
            "x": round(x, 2),
            "y": round(y, 2),
            "value": round(pr, 4),
            "symbolSize": round(size, 1),
            "category": cat_idx,
        }
        if item_style:
            item["itemStyle"] = item_style
        node_items.append(item)

    seen_edges: set[frozenset[str]] = set()
    link_items: list[dict] = []
    for source, targets in edges.items():
        if source not in node_set:
            continue
        for target in targets:
            pair = frozenset((source, target))
            if target in node_set and source != target and pair not in seen_edges:
                is_incident = selected_node in (source, target) if selected_node else False
                link_opacity = 0.85 if is_incident else (0.12 if selected_node else 0.4)
                link_width = 2.4 if is_incident else 1.2
                link_items.append(
                    {
                        "source": source,
                        "target": target,
                        "lineStyle": {"opacity": link_opacity, "width": link_width},
                    }
                )
                seen_edges.add(pair)

    axis_common = {
        "type": "value",
        "min": -1,
        "max": 1,
        "silent": True,
        "axisLabel": {"show": False},
        "axisTick": {"show": False},
        "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.14)"}},
        "splitLine": {
            "show": True,
            "lineStyle": {"color": "rgba(255,255,255,0.07)", "type": "dashed"},
        },
    }

    return {
        "tooltip": {"trigger": "item"},
        "grid": {"top": 52, "right": 16, "bottom": 16, "left": 16},
        "xAxis": axis_common,
        "yAxis": axis_common,
        "legend": [
            {
                "data": [cat["name"] for cat in categories],
                "top": 0,
                "textStyle": {"fontSize": 11},
            }
        ],
        "series": [
            {
                "name": "Concept Space",
                "type": "graph",
                "layout": "none",
                "roam": True,
                "top": 56,
                "selectedMode": "single",
                "data": node_items,
                "links": link_items,
                "categories": categories,
                "label": {"show": True, "position": "right", "fontSize": 11},
                "labelLayout": {"hideOverlap": True},
                "lineStyle": {"color": "source", "curveness": 0.15},
                "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3, "opacity": 0.95}},
                "select": {"itemStyle": {"borderColor": "#ffffff", "borderWidth": 4}},
            }
        ],
    }


def get_cache_stats() -> dict[str, int]:
    """Return count and approximate size (in KB) of cached Wikipedia pages on disk."""
    cache_dir = scraper.DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return {"count": 0, "size_kb": 0}
    files = list(cache_dir.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"count": len(files), "size_kb": round(total_bytes / 1024)}


def run_exploration(
    wiki_url: str,
    top_k: int,
    min_in_degree: int,
    method: str,
    scope: str,
    summary_scale: float,
    num_topics: int,
    extra_stopwords: str = "",
    *,
    alpha: str | float = "symmetric",
    eta: str | float | None = None,
    no_below: int = 2,
    no_above: float = 0.85,
) -> dict:
    """Run the full exploration pipeline: scraper, summarizer, topic modeling, and evaluation."""
    seed_url, seed_title = validate_wikipedia_url(wiki_url)

    neighborhood = scraper.discover_neighborhood(
        seed_url,
        top_k=top_k,
        min_in_degree=min_in_degree,
        min_nodes=min(5, top_k),
    )
    if not neighborhood.candidates:
        raise ValueError("No related concepts were found in this Wikipedia article.")

    articles = scraper.fetch_neighborhood_documents(neighborhood)
    docs = {node_id: article.full for node_id, article in articles.items() if article.full.strip()}
    if not docs:
        raise ValueError(
            "Unable to extract article content. Please check your network connection or cache."
        )

    seed_id = urlparse(seed_url).path
    if seed_id not in docs:
        raise ValueError(
            "Unable to fetch the seed article. Please verify the Wikipedia URL and try again."
        )

    candidate_labels = {candidate.href: candidate.text for candidate in neighborhood.candidates}
    labels = {
        node_id: candidate_labels.get(
            node_id, unquote(node_id.rsplit("/", 1)[-1]).replace("_", " ")
        )
        for node_id in docs
    }
    labels[seed_id] = seed_title

    summaries: dict[str, summarizer.NodeSummary] = {}
    node_rouge: dict[str, evaluation.DocumentRougeResult | None] = {}
    for node_id in docs:
        article = articles[node_id]
        n_sents, _ = compute_target_sentences(article.lead, article.full, summary_scale)
        node_summary = summarizer.summarize_node(
            article.lead,
            article.body,
            method=method,
            scope=scope,
            n_sentences=n_sents,
        )
        summaries[node_id] = node_summary
        node_rouge[node_id] = (
            evaluation.compute_rouge_score(node_summary.text, article.lead)
            if article.lead.strip() and node_summary.text.strip()
            else None
        )

    clustering = None
    clustering_warning = None
    stopword_set = {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}

    if len(docs) >= 2:
        try:
            clustering = topic_modeling.cluster_topic_nodes(
                docs,
                num_topics=min(num_topics, len(docs)),
                no_below=no_below,
                no_above=no_above,
                alpha=alpha,
                eta=eta,
                extra_stopwords=stopword_set if stopword_set else None,
            )
        except ValueError as exc:
            clustering_warning = str(exc)

    assignments = clustering.assignments if clustering else {}
    visible_nodes = list(docs)
    visible_edges = {
        source: targets & set(visible_nodes)
        for source, targets in neighborhood.edges.items()
        if source in visible_nodes
    }

    topology = evaluation.compute_graph_topology(visible_edges)

    topic_coherence = None
    if clustering:
        try:
            topic_coherence = evaluation.compute_topic_coherence(clustering)
        except Exception:
            topic_coherence = None

    section_keywords: dict[str, dict[str, list[str]]] = {}
    for node_id in visible_nodes:
        try:
            cached_soup = scraper.fetch_wikipedia_page(f"{scraper.WIKIPEDIA_BASE_URL}{node_id}")
            body_tag = scraper.get_body_content(cached_soup)
            if body_tag:
                section_keywords[node_id] = scraper.extract_section_keywords(body_tag)
        except (requests.RequestException, ValueError):
            section_keywords[node_id] = {}

    rouge_eval = None
    try:
        hypotheses = {nid: s.text for nid, s in summaries.items() if s.text.strip()}
        references = {nid: articles[nid].lead for nid in hypotheses if articles[nid].lead.strip()}
        matched_keys = set(hypotheses.keys()) & set(references.keys())
        if matched_keys:
            rouge_eval = evaluation.evaluate_summaries(
                {k: hypotheses[k] for k in matched_keys},
                {k: references[k] for k in matched_keys},
            )
    except Exception:
        rouge_eval = None

    return {
        "seed_url": seed_url,
        "seed_id": seed_id,
        "seed_title": seed_title,
        "neighborhood": neighborhood,
        "nodes": visible_nodes,
        "edges": visible_edges,
        "labels": labels,
        "articles": articles,
        "summaries": summaries,
        "node_rouge": node_rouge,
        "clustering": clustering,
        "assignments": assignments,
        "clustering_warning": clustering_warning,
        "topology": topology,
        "topic_coherence": topic_coherence,
        "section_keywords": section_keywords,
        "rouge_eval": rouge_eval,
    }


def run_single_benchmark(lead: str, body: str, summary_scale: float = 0.5) -> list[dict]:
    """Run extractive summarizer variants on a single article and compute latency & ROUGE scores."""
    full = "\n\n".join(part for part in (lead, body) if part.strip())
    n_sents, _ = compute_target_sentences(lead, full, summary_scale)
    variants = [
        ("TextRank (Full)", "textrank", "full"),
        ("TextRank (Lead)", "textrank", "lead"),
        ("LexRank (Full)", "lexrank", "full"),
        ("LexRank (Lead)", "lexrank", "lead"),
    ]

    results = []
    for label, method, scope in variants:
        t0 = time.perf_counter()
        summary = summarizer.summarize_node(
            lead, body, method=method, scope=scope, n_sentences=n_sents
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rouge_metrics = evaluation.compute_rouge_score(summary.text, lead) if lead.strip() else None

        results.append(
            {
                "label": label,
                "method": method,
                "scope": scope,
                "text": summary.text,
                "elapsed_ms": elapsed_ms,
                "rouge1_f1": rouge_metrics.rouge1.fmeasure if rouge_metrics else 0.0,
                "rouge2_f1": rouge_metrics.rouge2.fmeasure if rouge_metrics else 0.0,
                "rougeL_f1": rouge_metrics.rougeL.fmeasure if rouge_metrics else 0.0,
            }
        )
    return results
