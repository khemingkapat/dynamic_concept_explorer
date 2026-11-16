"""Dynamic Concept Explorer: Visualizing and Summarizing Wikipedia Topic Networks.

A full-featured Streamlit dashboard integrating all components in the repository:
1. scraper.py: 2-pass candidate outbound ranking, in-degree pruning, section keyword extraction, caching.
2. summarizer.py: Extractive TextRank, LexRank, and neural BART summarization with scope tuning.
3. topic_modeling.py: Unsupervised LDA topic discovery, node clustering, and keyword weights.
4. visualization.py: Section WordClouds and NetworkX word co-occurrence graphs.
5. evaluation.py: Graph topology metrics, Gensim C_v topic coherence, and ROUGE-1/2/L evaluation.
6. scripts/compare_node_summaries.py: Benchmarking summarizers side-by-side with latency and ROUGE.
"""

from __future__ import annotations

import html
import json
import re
import time
from urllib.parse import quote, unquote, urlparse

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import streamlit as st

from dynamic_concept_explorer import evaluation, scraper, summarizer, topic_modeling, visualization

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

PRESET_TOPICS = [
    "Machine learning",
    "Artificial intelligence",
    "Photosynthesis",
    "French Revolution",
    "General relativity",
    "Game theory",
    "Natural language processing",
    "Deep learning",
]


def _seed_details(value: str) -> tuple[str, str]:
    """Return a canonical Wikipedia URL and human-readable title for a topic or URL."""
    value = value.strip()
    if not value:
        raise ValueError("กรุณาระบุหัวข้อหรือ URL ของ Wikipedia")

    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        hostname = (parsed.hostname or "").lower()
        is_wikipedia_host = hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")
        if not is_wikipedia_host or not parsed.path.startswith("/wiki/"):
            raise ValueError(
                "รองรับเฉพาะ URL บทความจาก Wikipedia (เช่น https://en.wikipedia.org/wiki/...)"
            )
        url = f"https://{hostname}{parsed.path}"
    else:
        title = re.sub(r"\s+", "_", value)
        url = f"https://en.wikipedia.org/wiki/{quote(title, safe='()_')}"

    slug = urlparse(url).path.removeprefix("/wiki/")
    return url, unquote(slug).replace("_", " ")


def _dot_graph(
    seed_id: str,
    nodes: list[str],
    edges: dict[str, set[str]],
    assignments: dict[str, topic_modeling.NodeTopicAssignment],
    pagerank_scores: dict[str, float] | None = None,
) -> str:
    """Create a Graphviz DOT network representation with LDA topic colors and PageRank sizing."""
    lines = [
        "graph {",
        '  graph [bgcolor="transparent", overlap=false, splines=true, pad="0.3"];',
        '  node [style="filled", fontname="Arial", fontsize=11, fontcolor="#f4f3fa", '
        'color="#39364d", penwidth=1.5, margin="0.18,0.12"];',
        '  edge [color="#514c67", penwidth=1.2];',
    ]

    node_keys = {node: f"n{index}" for index, node in enumerate(nodes)}
    max_pr = max((pagerank_scores or {}).values(), default=1.0) or 1.0

    for node in nodes:
        assignment = assignments.get(node)
        color = TOPIC_COLORS[assignment.topic_id % len(TOPIC_COLORS)] if assignment else "#9b8cff"
        label = unquote(node.rsplit("/", 1)[-1]).replace("_", " ")
        is_seed = node == seed_id
        shape = "doublecircle" if is_seed else "ellipse"
        border_color = "#ffd166" if is_seed else "#39364d"
        penwidth = 3.0 if is_seed else 1.5

        # Degree / PageRank relative sizing
        pr = (pagerank_scores or {}).get(node, 0.0)
        scale = 0.8 + (1.2 * (pr / max_pr)) if pagerank_scores else 1.0
        width = f"{0.7 * scale:.2f}"
        height = f"{0.4 * scale:.2f}"

        lines.append(
            f"  {node_keys[node]} [label={json.dumps(label, ensure_ascii=False)}, "
            f"fillcolor={json.dumps(color)}, "
            f"color={json.dumps(border_color)}, "
            f"penwidth={penwidth}, "
            f"width={width}, height={height}, "
            f"shape={shape}];"
        )

    seen_edges: set[frozenset[str]] = set()
    for source, targets in edges.items():
        if source not in node_keys:
            continue
        for target in targets:
            edge = frozenset((source, target))
            if target in node_keys and source != target and edge not in seen_edges:
                lines.append(f"  {node_keys[source]} -- {node_keys[target]};")
                seen_edges.add(edge)

    lines.append("}")
    return "\n".join(lines)


def _get_cache_stats() -> dict[str, int]:
    """Return count and approximate size of cached pages on disk."""
    cache_dir = scraper.DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return {"count": 0, "size_kb": 0}
    files = list(cache_dir.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"count": len(files), "size_kb": round(total_bytes / 1024)}


def _explore(
    topic: str,
    top_k: int,
    min_in_degree: int,
    method: str,
    scope: str,
    n_sentences: int,
    num_topics: int,
    extra_stopwords: str = "",
) -> dict:
    """Run full exploration: scraper, summarizer, topic modeling, and evaluation."""
    seed_url, seed_title = _seed_details(topic)

    # 1. Scraper Module: 2-Pass Candidate Outbound Ranking & In-degree Pruning
    neighborhood = scraper.discover_neighborhood(
        seed_url,
        top_k=top_k,
        min_in_degree=min_in_degree,
        min_nodes=min(5, top_k),
    )
    if not neighborhood.candidates:
        raise ValueError("ไม่พบหัวข้อที่เกี่ยวข้องในบทความนี้")

    # Fetch neighborhood articles (lead & body)
    articles = scraper.fetch_neighborhood_documents(neighborhood)
    docs = {node_id: article.full for node_id, article in articles.items() if article.full.strip()}
    if not docs:
        raise ValueError("ดึงเนื้อหาบทความไม่ได้ ลองตรวจสอบการเชื่อมต่ออินเทอร์เน็ตหรือแคช")

    seed_id = urlparse(seed_url).path
    if seed_id not in docs:
        raise ValueError("ดึงบทความหัวข้อเริ่มต้นไม่ได้ ลองตรวจสอบชื่อหัวข้อแล้วค้นหาใหม่อีกครั้ง")

    candidate_labels = {candidate.href: candidate.text for candidate in neighborhood.candidates}
    labels = {
        node_id: candidate_labels.get(
            node_id, unquote(node_id.rsplit("/", 1)[-1]).replace("_", " ")
        )
        for node_id in docs
    }
    labels[seed_id] = seed_title

    # 2. Summarizer Module: Node Overviews (TextRank, LexRank, or BART)
    summaries = {
        node_id: summarizer.summarize_node(
            articles[node_id].lead,
            articles[node_id].body,
            method=method,
            scope=scope,
            n_sentences=n_sentences,
        )
        for node_id in docs
    }

    # 3. Topic Modeling Module: Unsupervised LDA Topic Discovery
    clustering = None
    clustering_warning = None
    stopword_set = {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}

    if len(docs) >= 2:
        try:
            clustering = topic_modeling.cluster_topic_nodes(
                docs,
                num_topics=min(num_topics, len(docs)),
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

    # 4. Evaluation Module: Graph Topology Metrics
    topology = evaluation.compute_graph_topology(visible_edges)

    # 5. Evaluation Module: LDA Topic Coherence (C_v)
    topic_coherence = None
    if clustering:
        try:
            topic_coherence = evaluation.compute_topic_coherence(clustering)
        except Exception:
            topic_coherence = None

    # 6. Scraper Module: Section Keywords Extraction for all nodes
    section_keywords: dict[str, dict[str, list[str]]] = {}
    for node_id in visible_nodes:
        try:
            cached_soup = scraper.fetch_wikipedia_page(f"{scraper.WIKIPEDIA_BASE_URL}{node_id}")
            body_tag = scraper.get_body_content(cached_soup)
            if body_tag:
                sec_kw = scraper.extract_section_keywords(body_tag)
                section_keywords[node_id] = sec_kw
        except Exception:
            section_keywords[node_id] = {}

    # 7. Evaluation Module: Macro ROUGE Summaries evaluation against Wikipedia lead abstracts
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
        "seed_id": seed_id,
        "seed_title": seed_title,
        "neighborhood": neighborhood,
        "nodes": visible_nodes,
        "edges": visible_edges,
        "labels": labels,
        "articles": articles,
        "summaries": summaries,
        "clustering": clustering,
        "assignments": assignments,
        "clustering_warning": clustering_warning,
        "topology": topology,
        "topic_coherence": topic_coherence,
        "section_keywords": section_keywords,
        "rouge_eval": rouge_eval,
    }


def _run_single_benchmark(
    lead: str,
    body: str,
    use_bart: bool = False,
) -> list[dict]:
    """Run all summarizer variants on a single article and compute latency & ROUGE scores."""
    variants = [
        ("TextRank (Lead)", "textrank", "lead"),
        ("TextRank (Full)", "textrank", "full"),
        ("LexRank (Full)", "lexrank", "full"),
    ]
    if use_bart:
        variants.append(("BART (Lead)", "bart", "lead"))

    results = []
    for label, method, scope in variants:
        t0 = time.perf_counter()
        summary = summarizer.summarize_node(lead, body, method=method, scope=scope, n_sentences=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rouge_metrics = evaluation.compute_rouge_score(summary.text, lead) if lead.strip() else None

        results.append(
            {
                "label": label,
                "method": method,
                "scope": scope,
                "text": summary.text,
                "elapsed_ms": elapsed_ms,
                "fallback_reason": summary.fallback_reason,
                "rouge1_f1": rouge_metrics.rouge1.fmeasure if rouge_metrics else 0.0,
                "rouge2_f1": rouge_metrics.rouge2.fmeasure if rouge_metrics else 0.0,
                "rougeL_f1": rouge_metrics.rougeL.fmeasure if rouge_metrics else 0.0,
                "rouge1_p": rouge_metrics.rouge1.precision if rouge_metrics else 0.0,
                "rouge1_r": rouge_metrics.rouge1.recall if rouge_metrics else 0.0,
            }
        )
    return results


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f4f3fa;
            --muted: #a19db2;
            --panel: #171622;
            --panel-border: #2c293a;
            --violet: #9b8cff;
            --violet-hover: #b3a9ff;
            --teal: #55d6be;
            --gold: #ffb454;
        }
        html, body, [class*="css"] {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .stApp {
            background: radial-gradient(ellipse at 80% -10%, #26203a 0%, #100f17 55%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #14131d;
            border-right: 1px solid var(--panel-border);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
        }
        .eyebrow {
            color: #b8adff;
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
        }
        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 3.8vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -.05em;
            line-height: 1.1;
            margin: .35rem 0 .6rem;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 720px;
        }
        .metric-card {
            background: #191824;
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            transition: transform 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #3f3a53;
        }
        .metric-label {
            color: var(--muted);
            font-size: .8rem;
            font-weight: 500;
            margin-bottom: 6px;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -.03em;
        }
        .metric-sub {
            color: #7d7890;
            font-size: .75rem;
            margin-top: 4px;
        }
        .section-card {
            background: #181723;
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 1.2rem;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0 0 4px;
        }
        .section-hint {
            color: var(--muted);
            font-size: .85rem;
            margin-bottom: 16px;
        }
        .badge {
            display: inline-block;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: .8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        div.stButton > button[kind="primary"] {
            background: var(--violet);
            border: 0;
            color: #141221;
            font-weight: 700;
            border-radius: 10px;
            min-height: 46px;
            transition: background 0.2s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--violet-hover);
            color: #141221;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            color: var(--muted);
        }
        .stTabs [aria-selected="true"] {
            background: #242235;
            color: var(--ink);
        }
        hr {
            border-color: var(--panel-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Dynamic Concept Explorer",
        page_icon="✳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_styles()

    # Session State Initialization
    if "topic_input" not in st.session_state:
        st.session_state["topic_input"] = "Machine learning"

    # --- SIDEBAR: Explorer Settings & Repository Modules Control ---
    with st.sidebar:
        st.markdown('<div class="eyebrow">EXPLORER DASHBOARD</div>', unsafe_allow_html=True)
        st.markdown("### ตั้งค่าระบบสำรวจความรู้")

        with st.form("explore_form"):
            topic = st.text_input(
                "หัวข้อหรือ Wikipedia URL",
                value=st.session_state["topic_input"],
                placeholder="เช่น Artificial intelligence, Photosynthesis",
                help="ใส่ชื่อหัวข้อภาษาอังกฤษ หรือ URL เต็มจาก en.wikipedia.org",
            )

            st.markdown("**1. โครงข่ายและตัวกรอง (Scraper & Pruning)**")
            top_k = st.slider(
                "จำนวนหัวข้อที่ค้นหา (Top-K)",
                min_value=5,
                max_value=25,
                value=10,
                help="จำนวน Candidate Links ที่คัดเลือกในรอบแรกโดย Scraper",
            )
            min_in_degree = st.slider(
                "ความเชื่อมโยงขั้นต่ำ (Min In-Degree)",
                min_value=0,
                max_value=5,
                value=1,
                help="คัดกรองหัวข้อด้วย 2-Pass Neighborhood In-Degree ป้องกันกราฟขยายตัวอย่างไร้ทิศทาง",
            )

            st.markdown("**2. การสรุปเนื้อหา (Summarizer Module)**")
            method = st.selectbox(
                "อัลกอริทึมสรุปผลเริ่มต้น",
                options=("textrank", "lexrank", "bart"),
                format_func={
                    "textrank": "TextRank (Extractive Graph)",
                    "lexrank": "LexRank (Corpus Centroid)",
                    "bart": "BART (Neural Abstractive)",
                }.get,
                help="เลือกอัลกอริทึมที่ใช้สร้าง Node Overviews อัตโนมัติ",
            )
            scope = st.selectbox(
                "ขอบเขตการอ่านเนื้อหา",
                options=("lead", "full"),
                format_func={"lead": "บทนำ (Lead Section)", "full": "ทั้งบทความ (Full Article)"}.get,
            )
            n_sentences = st.slider(
                "จำนวนประโยคสรุป (Node Overview)",
                min_value=1,
                max_value=5,
                value=2,
            )

            st.markdown("**3. การจัดกลุ่มหัวข้อ (Topic Modeling - LDA)**")
            num_topics = st.slider(
                "จำนวนกลุ่มหัวข้อ (LDA Topics)",
                min_value=2,
                max_value=6,
                value=3,
            )
            extra_stopwords = st.text_input(
                "คำหยุดพิเศษ (Custom Stopwords)",
                placeholder="เช่น also, use, model, study",
                help="คำหยุดภาษาอังกฤษเพิ่มเติม คั่นด้วยเครื่องหมายจุลภาค (,)",
            )

            submitted = st.form_submit_button(
                "✳ เริ่มการสำรวจแนวคิด",
                type="primary",
                use_container_width=True,
            )

        # Quick Preset Buttons
        st.markdown("**ตัวอย่างหัวข้อแนะนำ (Quick Presets):**")
        preset_cols = st.columns(2)
        for i, preset in enumerate(PRESET_TOPICS[:6]):
            col = preset_cols[i % 2]
            if col.button(preset, key=f"preset_{preset}", use_container_width=True):
                st.session_state["topic_input"] = preset
                st.session_state.pop("exploration", None)
                st.rerun()

        st.divider()

        # Cache Management Tool
        cache_stats = _get_cache_stats()
        st.markdown(
            f"**สถานะแคชบทความ:** `{cache_stats['count']} ไฟล์ (~{cache_stats['size_kb']} KB)`"
        )
        if st.button("🗑️ ล้างแคช Wikipedia (.cache/wiki)", use_container_width=True):
            scraper.clear_page_cache()
            st.success("ล้างแคชเรียบร้อยแล้ว!")
            st.rerun()

        st.caption(
            "ระบบรวบรวม 5 โมดูล: Scraper, Summarizer, Topic Modeling, Visualization และ Evaluation"
        )

    # --- MAIN VIEW HEADER ---
    st.markdown(
        '<div class="eyebrow">DYNAMIC CONCEPT EXPLORER · NLP & GRAPH KNOWLEDGE PIPELINE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">ระบบสำรวจโครงข่ายความรู้<br>และวิเคราะห์สรุปแนวคิดอัตโนมัติ</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">เครื่องมือสำรวจโครงข่ายแนวคิดและวิเคราะห์เนื้อหาบทความ Wikipedia แบบอินเทอร์แอคทีฟ '
        "ผสานการตัดทอนเนื้อหา (TextRank / LexRank / BART), การจัดกลุ่มความหมายด้วย LDA Topic Modeling, "
        "การสร้างภาพเวิร์ดคลาวด์และเครือข่ายคำร่วม (NetworkX / WordCloud), "
        "พร้อมการประเมินโครงสร้างกราฟและมาตรวัด ROUGE / C_v Coherence เชิงปริมาณ</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Handle Form Submission or Preset Trigger
    if submitted or ("exploration" not in st.session_state and st.session_state.get("topic_input")):
        st.session_state.pop("exploration", None)
        active_topic = topic if submitted else st.session_state["topic_input"]
        try:
            with st.spinner(f"กำลังสำรวจและวิเคราะห์หัวข้อ '{active_topic}' จาก Wikipedia..."):
                st.session_state["exploration"] = _explore(
                    active_topic,
                    top_k,
                    min_in_degree,
                    method,
                    scope,
                    n_sentences,
                    num_topics,
                    extra_stopwords,
                )
        except (ValueError, requests.RequestException) as exc:
            st.error(f"สำรวจไม่สำเร็จ: {exc}")

    result = st.session_state.get("exploration")
    if not result:
        st.info("👈 กรุณาระบุหัวข้อหรือเลือกตัวอย่างที่แถบด้านซ้าย แล้วกดปุ่ม **เริ่มการสำรวจแนวคิด**")
        return

    # --- TOP KPI METRICS BAR ---
    topic_labels = result["labels"]
    assignments = result["assignments"]
    clustering = result["clustering"]
    topology: evaluation.GraphTopologyMetrics = result["topology"]
    coherence: evaluation.TopicCoherenceResult | None = result["topic_coherence"]
    seed_title = html.escape(result["seed_title"])

    st.markdown(f"### แผนที่แนวคิดสำหรับ: **{seed_title}**")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpis = [
        ("แนวคิดที่ค้นพบ", str(topology.node_count), "Discovered Nodes"),
        ("ความเชื่อมโยง", str(topology.edge_count), "Network Edges"),
        ("กลุ่มความรู้ (LDA)", str(clustering.num_topics if clustering else 0), "Topic Clusters"),
        ("ความหนาแน่นกราฟ", f"{topology.density:.3f}", "Graph Density"),
        ("Clustering Coeff", f"{topology.average_clustering:.3f}", "Avg Clustering"),
        (
            "Topic Coherence",
            f"{coherence.mean_coherence:.3f}" if coherence else "N/A",
            "Mean C_v Score",
        ),
    ]
    for col, (label, val, sub) in zip([kpi1, kpi2, kpi3, kpi4, kpi5, kpi6], kpis, strict=True):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # --- TAB NAVIGATION ---
    tab_graph, tab_deepdive, tab_vis, tab_benchmark, tab_eval = st.tabs(
        [
            "🌐 โครงข่ายแนวคิดและกราฟ (Concept Network)",
            "📖 เจาะลึกแนวคิด & แยกส่วนเนื้อหา (Concept Inspector)",
            "☁️ การแสดงผลภาพ (WordCloud & Co-occurrence)",
            "⚖️ เปรียบเทียบโมเดลสรุปผล (Summarizer Benchmark)",
            "📊 การประเมินผลเชิงปริมาณ (Quantitative Evaluation)",
        ]
    )

    # =========================================================================
    # TAB 1: CONCEPT NETWORK & GRAPH TOPOLOGY
    # =========================================================================
    with tab_graph:
        col_graph_view, col_graph_info = st.columns([1.5, 1], gap="large")

        with col_graph_view:
            st.markdown(
                '<div class="section-title">แผนภาพเครือข่ายความสัมพันธ์ (Concept Graph)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-hint">โหนดแทนแนวคิด · สีตามกลุ่ม LDA Topic · '
                "ขนาดโหนดถ่วงน้ำหนักตาม PageRank · วงกลมซ้อนสีทองคือหัวข้อเริ่มต้น</div>",
                unsafe_allow_html=True,
            )

            # Render Graphviz DOT Network
            dot_code = _dot_graph(
                result["seed_id"],
                result["nodes"],
                result["edges"],
                assignments,
                pagerank_scores=topology.pagerank,
            )
            with st.container(border=True):
                st.graphviz_chart(dot_code, use_container_width=True)

            # Topic Cluster Chips
            if clustering:
                st.markdown("**กลุ่มความรู้และคำสำคัญเด่น (LDA Topic Clusters):**")
                chips_html = " ".join(
                    f'<span class="badge" style="background:{TOPIC_COLORS[tid % len(TOPIC_COLORS)]};color:#12101e">'
                    f"กลุ่ม {tid + 1}: {desc.label}</span>"
                    for tid, desc in clustering.topics.items()
                )
                st.markdown(chips_html, unsafe_allow_html=True)

            if result["clustering_warning"]:
                st.warning(f"หมายเหตุการจัดกลุ่ม: {result['clustering_warning']}")

        with col_graph_info:
            st.markdown(
                '<div class="section-title">ลำดับความสำคัญของแนวคิด (Centrality Leaderboard)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-hint">จัดอันดับแนวคิดที่มีอิทธิพลสูงที่สุดในโครงข่ายตาม PageRank และ Centrality</div>',
                unsafe_allow_html=True,
            )

            # Top Centrality Nodes Table
            centrality_data = []
            for node_id in result["nodes"]:
                assignment = assignments.get(node_id)
                cluster_label = f"กลุ่ม {assignment.topic_id + 1}" if assignment else "-"
                centrality_data.append(
                    {
                        "แนวคิด (Concept)": topic_labels[node_id],
                        "กลุ่ม LDA": cluster_label,
                        "PageRank": round(topology.pagerank.get(node_id, 0.0), 4),
                        "In-Degree Centrality": round(
                            topology.in_degree_centrality.get(node_id, 0.0), 3
                        ),
                        "Betweenness": round(topology.betweenness_centrality.get(node_id, 0.0), 4),
                    }
                )
            df_centrality = pd.DataFrame(centrality_data).sort_values("PageRank", ascending=False)
            st.dataframe(df_centrality, hide_index=True, use_container_width=True, height=280)

            # 2-Pass Discovery & Pruning Breakdown
            st.markdown(
                '<div class="section-title" style="margin-top:1rem;">กระบวนการคัดกรอง 2-Pass Pruning</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="section-hint">คัดกรองจาก 후보ทั้งหมด {len(result["neighborhood"].candidates) + len(result["neighborhood"].pruned)} รายการ '
                f"ผ่านเกณฑ์ In-Degree ≥ {min_in_degree} จำนวน {len(result['neighborhood'].candidates)} รายการ</div>",
                unsafe_allow_html=True,
            )

            with st.expander("🔍 ดูรายการหัวข้อที่ผ่านการคัดเลือก (Candidate Links)"):
                candidate_rows = [
                    {
                        "ชื่อหัวข้อ": c.text,
                        "คะแนน Scorer": round(c.score, 3),
                        "ความถี่ในบทความ": c.count,
                        "อยู่ในบทนำ": "✓" if c.in_lead else "—",
                        "In-Degree ในเครือข่าย": c.in_degree,
                    }
                    for c in result["neighborhood"].candidates
                ]
                st.dataframe(
                    pd.DataFrame(candidate_rows), hide_index=True, use_container_width=True
                )

            if result["neighborhood"].pruned:
                with st.expander(
                    f"✂️ ดูรายการที่ถูกคัดออก ({len(result['neighborhood'].pruned)} รายการที่ไม่ผ่าน In-Degree)"
                ):
                    pruned_rows = [
                        {
                            "ชื่อหัวข้อ": p.text,
                            "In-Degree": p.in_degree,
                            "คะแนนเบื้องต้น": round(p.score, 3),
                        }
                        for p in result["neighborhood"].pruned
                    ]
                    st.dataframe(
                        pd.DataFrame(pruned_rows), hide_index=True, use_container_width=True
                    )

    # =========================================================================
    # TAB 2: CONCEPT INSPECTOR & SECTIONS
    # =========================================================================
    with tab_deepdive:
        col_select, col_actions = st.columns([2, 1])
        with col_select:
            selected_node = st.selectbox(
                "เลือกแนวคิดที่ต้องการเจาะลึก",
                options=result["nodes"],
                format_func=lambda n: (
                    f"{topic_labels[n]} ({'หัวข้อตั้งต้น' if n == result['seed_id'] else n})"
                ),
                label_visibility="visible",
            )
        with col_actions:
            st.write("")
            st.write("")
            st.link_button(
                "🌐 เปิดอ่านบน Wikipedia ↗",
                f"https://en.wikipedia.org{selected_node}",
                use_container_width=True,
            )

        article = result["articles"][selected_node]
        summary = result["summaries"][selected_node]
        assignment = assignments.get(selected_node)
        node_sections = result["section_keywords"].get(selected_node, {})

        c_left, c_right = st.columns([1.2, 1], gap="large")

        with c_left:
            st.markdown(f"## {topic_labels[selected_node]}")

            # Topic Assignment & Probability Distribution
            if assignment and clustering:
                desc = clustering.topics[assignment.topic_id]
                color = TOPIC_COLORS[assignment.topic_id % len(TOPIC_COLORS)]
                st.markdown(
                    f'<span class="badge" style="background:{color};color:#12101e;font-size:.88rem">'
                    f"กลุ่ม {assignment.topic_id + 1} · {desc.label}</span> "
                    f'<span style="color:var(--muted);font-size:.85rem">ความเชื่อมั่น (Confidence): {assignment.confidence * 100:.1f}%</span>',
                    unsafe_allow_html=True,
                )

                # Distribution Progress Bars
                with st.expander("📊 ดูการกระจายความน่าจะเป็นในทุกกลุ่ม (LDA Topic Distribution)"):
                    for tid, prob in sorted(assignment.distribution, key=lambda x: x[0]):
                        t_label = clustering.topics[tid].label
                        st.caption(f"กลุ่ม {tid + 1} ({t_label[:30]}...): {prob * 100:.1f}%")
                        st.progress(float(prob))

            st.write("")

            # Summary Card
            st.markdown(
                '<div class="section-title">สรุปเนื้อหาที่ผ่านการกลั่นกรอง (Distilled Overview)</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                if summary.text:
                    st.markdown(f"**{summary.text}**")
                else:
                    st.warning("บทความนี้มีข้อความสั้นเกินไปหรือไม่พบประโยคที่มีความยาวตามเกณฑ์")

                st.divider()
                st.caption(
                    f"⚙️ โมเดล: **{summary.method.title()}** · ขอบเขต: **{summary.scope.title()}** · "
                    f"ใช้เวลา: **{summary.elapsed_ms:.1f} ms** · ประโยค: **{len(summary.sentences)} ประโยค**"
                )
                if summary.fallback_reason:
                    st.caption("ℹ️ มีการใช้ TextRank ทดแทนเนื่องจากเหตุผลทางทรัพยากร (Fallback)")

            # Live Interactive Re-summarize Controls
            with st.expander("⚡ ทดลองเปลี่ยนอัลกอริทึมสรุปผลสำหรับแนวคิดนี้แบบเรียลไทม์"):
                r_col1, r_col2, r_col3 = st.columns(3)
                re_method = r_col1.selectbox(
                    "เลือกวิธีสรุปใหม่", ("textrank", "lexrank", "bart"), index=0, key="re_m"
                )
                re_scope = r_col2.selectbox("ขอบเขตข้อความ", ("lead", "full"), index=0, key="re_s")
                re_n = r_col3.slider("จำนวนประโยค", 1, 4, 2, key="re_n")
                if st.button("ประมวลผลสรุปใหม่ทันที", use_container_width=True):
                    t_start = time.perf_counter()
                    new_sum = summarizer.summarize_node(
                        article.lead,
                        article.body,
                        method=re_method,
                        scope=re_scope,
                        n_sentences=re_n,
                    )
                    t_el = (time.perf_counter() - t_start) * 1000.0
                    st.success(f"ผลลัพธ์ ({new_sum.method.title()} - {t_el:.1f} ms):")
                    st.write(new_sum.text)

            # Full Article Prose Expander
            with st.expander("📜 ดูข้อความเต็มของบทความ (Full Article Text)"):
                st.text_area("Prose Content", value=article.full, height=300, disabled=True)

        with c_right:
            st.markdown(
                '<div class="section-title">บทนำต้นฉบับจาก Wikipedia (Lead Abstract)</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.write(article.lead if article.lead else "(ไม่มีข้อความบทนำ)")

            st.write("")
            st.markdown(
                '<div class="section-title">หัวข้อย่อยและคำสำคัญ (Section Breakdown & Keywords)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-hint">สกัดจาก scraper.extract_section_keywords เพื่อแสดงโครงสร้างประเด็นในบทความ</div>',
                unsafe_allow_html=True,
            )

            if node_sections:
                for sec_title, kw_list in list(node_sections.items())[:8]:
                    clean_title = re.sub(r"\[edit\]", "", sec_title).strip()
                    with st.expander(f"📁 {clean_title} ({len(kw_list)} คำสำคัญ)"):
                        if kw_list:
                            chips = " ".join(
                                f'<span style="background:#282537;color:#d3d0e2;border-radius:12px;padding:3px 8px;font-size:.78rem;margin:2px;display:inline-block">'
                                f"{html.escape(k)}</span>"
                                for k in kw_list[:25]
                            )
                            st.markdown(chips, unsafe_allow_html=True)
                        else:
                            st.caption("ไม่มีลิงก์คำสำคัญในส่วนนี้")
            else:
                st.info("ไม่พบโครงสร้าง <section> ย่อยในบทความนี้")

    # =========================================================================
    # TAB 3: VISUALIZATIONS (WORDCLOUD & CO-OCCURRENCE)
    # =========================================================================
    with tab_vis:
        st.markdown(
            '<div class="section-title">การแสดงผลภาพขั้นสูง (Advanced Visualizations)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-hint">เชื่อมโยงฟังก์ชันจาก visualization.py สำหรับการสร้าง WordCloud ประจำส่วน และ กราฟคำร่วม (Word Co-occurrence Network)</div>',
            unsafe_allow_html=True,
        )

        vis_node = st.selectbox(
            "เลือกแนวคิดที่ต้องการสร้างแผนภาพ",
            options=result["nodes"],
            format_func=lambda n: topic_labels[n],
            key="vis_node_select",
        )

        v_tab_wc, v_tab_cooc = st.tabs(
            [
                "☁️ เวิร์ดคลาวด์ประจำบทความ (Section WordCloud)",
                "🕸️ กราฟความสัมพันธ์คำร่วม (Word Co-occurrence Network)",
            ]
        )

        with v_tab_wc:
            sec_dict = result["section_keywords"].get(vis_node, {})
            # Aggregate all section keywords
            all_keywords = [kw for kws in sec_dict.values() for kw in kws]

            wc_col1, wc_col2 = st.columns([1, 2])
            with wc_col1:
                st.markdown("**การตั้งค่า WordCloud**")
                wc_source = st.radio(
                    "แหล่งที่มาของคำสำคัญ",
                    ["คำสำคัญรวมทั้งบทความ (All Keywords)", "เลือกตามหัวข้อย่อย (Specific Section)"],
                )
                selected_sec = ""
                if wc_source == "เลือกตามหัวข้อย่อย (Specific Section)" and sec_dict:
                    selected_sec = st.selectbox("เลือกหัวข้อย่อย", list(sec_dict.keys()))
                    kw_target = sec_dict[selected_sec]
                else:
                    kw_target = all_keywords

                bg_color = st.selectbox("สีพื้นหลัง", ["white", "black", "#171622"])
                btn_gen_wc = st.button("🎨 สร้าง WordCloud", use_container_width=True)

            with wc_col2:
                if btn_gen_wc or kw_target:
                    if kw_target:
                        with st.spinner("กำลังเรนเดอร์ WordCloud..."):
                            wc = visualization.generate_section_wordcloud(
                                kw_target,
                                section_title=selected_sec or f"Keywords: {topic_labels[vis_node]}",
                                background_color=bg_color,
                                show=False,
                            )
                            if wc:
                                st.image(
                                    wc.to_image(),
                                    caption=f"WordCloud - {topic_labels[vis_node]}",
                                    use_container_width=True,
                                )
                                plt.close("all")
                            else:
                                st.warning("คำสำคัญไม่เพียงพอสำหรับการสร้าง WordCloud")
                    else:
                        st.info("ไม่พบรายการคำสำคัญสำหรับโหนดนี้ ลองเลือกโหนดอื่น")

        with v_tab_cooc:
            st.markdown(
                '<div class="section-hint">สร้างโครงข่ายคำร่วม (Pairwise Word Co-occurrence) โดยวัดความถี่การปรากฏร่วมกัน '
                "และกำหนดขนาดโหนดตาม Degree Centrality (deg × 20)</div>",
                unsafe_allow_html=True,
            )
            cooc_col1, cooc_col2 = st.columns([1, 2])
            with cooc_col1:
                cooc_text_source = st.selectbox(
                    "ข้อความต้นทางสำหรับกราฟคำร่วม",
                    [
                        "สรุปใจความสำคัญ (Distilled Summary)",
                        "ประโยคแรกของบทนำ (First 2 Lead Sentences)",
                        "ข้อความกำหนดเอง (Custom Text)",
                    ],
                )
                if cooc_text_source == "สรุปใจความสำคัญ (Distilled Summary)":
                    raw_text = result["summaries"][vis_node].text
                elif cooc_text_source == "ประโยคแรกของบทนำ (First 2 Lead Sentences)":
                    sents = summarizer.split_sentences(result["articles"][vis_node].lead)
                    raw_text = " ".join(sents[:2])
                else:
                    raw_text = st.text_area(
                        "พิมพ์ข้อความที่ต้องการวิเคราะห์", value=result["summaries"][vis_node].text
                    )

                btn_gen_cooc = st.button("🕸️ เรนเดอร์กราฟคำร่วม", use_container_width=True)

            with cooc_col2:
                if btn_gen_cooc or raw_text:
                    if raw_text.strip():
                        # Extract substantive alphanumeric tokens (min 3 chars) to avoid punctuation nodes
                        tokens = re.findall(r"[a-zA-Z]{3,}", raw_text)
                        clean_str = " ".join(tokens)
                        if len(tokens) >= 3:
                            with st.spinner("กำลังคำนวณ Co-occurrence Graph..."):
                                G = visualization.build_cooccurrence_graph(clean_str)
                                fig, ax = plt.subplots(figsize=(9, 7))
                                pos = nx.spring_layout(G, seed=42)
                                node_sizes = [deg * 35 for deg in dict(G.degree()).values()]
                                nx.draw(
                                    G,
                                    pos,
                                    with_labels=True,
                                    font_size=9,
                                    node_size=node_sizes,
                                    font_color="#111",
                                    edge_color="#888",
                                    font_weight="bold",
                                    alpha=0.85,
                                    node_color="#55d6be",
                                    ax=ax,
                                )
                                ax.set_title(
                                    f"Word Co-occurrence: {topic_labels[vis_node]}",
                                    fontsize=12,
                                    fontweight="bold",
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.warning("ข้อความสั้นเกินไปสำหรับการสร้างกราฟคำร่วม")
                    else:
                        st.info("ไม่มีข้อความสำหรับประมวลผล")

    # =========================================================================
    # TAB 4: SUMMARIZER BENCHMARK LAB (compare_node_summaries.py)
    # =========================================================================
    with tab_benchmark:
        st.markdown(
            '<div class="section-title">ห้องทดลองเปรียบเทียบโมเดลสรุปผล (Summarization Benchmark Lab)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-hint">นำเข้าตรรกะการเปรียบเทียบจาก scripts/compare_node_summaries.py '
            "เพื่อวัดความเร็ว (Latency) และคะแนน ROUGE-1 / ROUGE-2 / ROUGE-L เทียบกับบทนำอ้างอิงของ Wikipedia</div>",
            unsafe_allow_html=True,
        )

        b_node = st.selectbox(
            "เลือกแนวคิดที่ต้องการทดสอบเปรียบเทียบ",
            options=result["nodes"],
            format_func=lambda n: topic_labels[n],
            key="benchmark_node_select",
        )
        b_use_bart = st.checkbox(
            "เปิดใช้งานโมเดล Neural Seq2Seq (BART) ร่วมด้วย (ดาวน์โหลดโมเดลขนาดใหญ่เมื่อเรียกใช้ครั้งแรก)",
            value=False,
        )

        if st.button("🚀 รันการเปรียบเทียบทุกโมเดล (Run Benchmark)", type="primary"):
            b_article = result["articles"][b_node]
            with st.spinner(f"กำลังเปรียบเทียบโมเดลสรุปผลสำหรับ '{topic_labels[b_node]}' ..."):
                bench_data = _run_single_benchmark(
                    b_article.lead, b_article.body, use_bart=b_use_bart
                )

            # Display comparative cards
            cols = st.columns(len(bench_data))
            for col, item in zip(cols, bench_data, strict=True):
                with col:
                    st.markdown(f"#### {item['label']}")
                    st.caption(f"⏱️ **{item['elapsed_ms']:.1f} ms**")
                    with st.container(border=True):
                        st.markdown(f"_{item['text']}_")
                    st.markdown(
                        f"**ROUGE F1-Scores:**<br>"
                        f"• R-1: `{item['rouge1_f1']:.3f}`<br>"
                        f"• R-2: `{item['rouge2_f1']:.3f}`<br>"
                        f"• R-L: `{item['rougeL_f1']:.3f}`",
                        unsafe_allow_html=True,
                    )

            st.write("")
            st.markdown("### ตารางเปรียบเทียบเชิงลึก (Metric Comparison Table)")
            df_bench = pd.DataFrame(
                [
                    {
                        "วิธีการสรุป (Method)": item["label"],
                        "เวลาประมวลผล (ms)": round(item["elapsed_ms"], 1),
                        "ROUGE-1 F1": round(item["rouge1_f1"], 4),
                        "ROUGE-2 F1": round(item["rouge2_f1"], 4),
                        "ROUGE-L F1": round(item["rougeL_f1"], 4),
                        "ROUGE-1 Precision": round(item["rouge1_p"], 4),
                        "ROUGE-1 Recall": round(item["rouge1_r"], 4),
                    }
                    for item in bench_data
                ]
            )
            st.dataframe(df_bench, hide_index=True, use_container_width=True)

    # =========================================================================
    # TAB 5: QUANTITATIVE EVALUATION & TOPIC COHERENCE (evaluation.py)
    # =========================================================================
    with tab_eval:
        st.markdown(
            '<div class="section-title">การประเมินผลเชิงปริมาณ (Quantitative Evaluation Dashboard)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-hint">รวมศูนย์การประเมินจาก evaluation.py: ความสอดคล้องของหัวข้อ (Gensim C_v), '
            "คุณสมบัติทางโทโพโลยีของโครงข่าย (Graph Topology), และมาตรวัด ROUGE เฉลี่ยทั้งชุดข้อมูล (Corpus Macro-Average)</div>",
            unsafe_allow_html=True,
        )

        e_col1, e_col2 = st.columns(2, gap="large")

        with e_col1:
            st.markdown("### 1. การประเมิน Topic Coherence ($C_v$)")
            if coherence and coherence.num_topics > 0:
                st.metric("Macro-Averaged C_v Coherence", f"{coherence.mean_coherence:.4f}")
                st.caption(
                    "เกณฑ์อ้างอิง: ค่า C_v > 0.4 แสดงว่าคำสำคัญในหัวข้อมีความสัมพันธ์และความหมายสอดคล้องกันสูง"
                )

                coh_rows = []
                for tid, c_score in sorted(coherence.topic_coherences.items()):
                    t_desc = clustering.topics.get(tid)
                    label = t_desc.label if t_desc else f"Topic {tid}"
                    coh_rows.append(
                        {
                            "กลุ่ม (Topic ID)": f"กลุ่ม {tid + 1}",
                            "คะแนน C_v": round(c_score, 4),
                            "คำสำคัญเด่น": label,
                        }
                    )
                st.dataframe(pd.DataFrame(coh_rows), hide_index=True, use_container_width=True)
            else:
                st.info("ไม่มีผลการประเมิน Topic Coherence (ต้องมีอย่างน้อย 2 บทความและคำศัพท์เพียงพอ)")

            st.write("")
            st.markdown("### 2. คุณสมบัติทางโทโพโลยีกราฟ (Graph Topology Metrics)")
            topo_rows = [
                {"มาตรวัด (Metric)": "จำนวนโหนด (Node Count)", "ค่า": str(topology.node_count)},
                {"มาตรวัด (Metric)": "จำนวนเส้นเชื่อม (Edge Count)", "ค่า": str(topology.edge_count)},
                {"มาตรวัด (Metric)": "ความหนาแน่นของกราฟ (Density)", "ค่า": f"{topology.density:.4f}"},
                {
                    "มาตรวัด (Metric)": "การแลกเปลี่ยนลิงก์สองทาง (Reciprocity)",
                    "ค่า": f"{topology.reciprocity:.4f}",
                },
                {
                    "มาตรวัด (Metric)": "สัมประสิทธิ์การรวมกลุ่มเฉลี่ย (Avg Clustering)",
                    "ค่า": f"{topology.average_clustering:.4f}",
                },
            ]
            st.dataframe(pd.DataFrame(topo_rows), hide_index=True, use_container_width=True)

        with e_col2:
            st.markdown("### 3. การประเมินสรุปผลภาพรวมทั้งโครงข่าย (Corpus ROUGE Evaluation)")
            rouge_eval = result.get("rouge_eval")
            if rouge_eval:
                r1_col, r2_col, rl_col = st.columns(3)
                r1_col.metric("Mean ROUGE-1 F1", f"{rouge_eval.mean_rouge1.fmeasure:.4f}")
                r2_col.metric("Mean ROUGE-2 F1", f"{rouge_eval.mean_rouge2.fmeasure:.4f}")
                rl_col.metric("Mean ROUGE-L F1", f"{rouge_eval.mean_rougeL.fmeasure:.4f}")

                st.markdown("**ผลการประเมินรายบทความ (Per-Document ROUGE):**")
                per_doc_rows = [
                    {
                        "หัวข้อบทความ": topic_labels.get(doc_id, doc_id),
                        "ROUGE-1 F1": round(res.rouge1.fmeasure, 3),
                        "ROUGE-2 F1": round(res.rouge2.fmeasure, 3),
                        "ROUGE-L F1": round(res.rougeL.fmeasure, 3),
                        "Precision": round(res.rouge1.precision, 3),
                        "Recall": round(res.rouge1.recall, 3),
                    }
                    for doc_id, res in rouge_eval.per_document.items()
                ]
                st.dataframe(
                    pd.DataFrame(per_doc_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=280,
                )
            else:
                st.info("ไม่สามารถคำนวณ Macro ROUGE ได้เนื่องจากไม่มีบทนำที่สมบูรณ์เพียงพอ")

            st.write("")
            st.markdown("### 4. สถาปัตยกรรมระบบ (System Architecture Pipeline)")
            st.code(
                """
Seed Topic -> Scraper (2-Pass Pruning & Section Keywords)
                 │
                 ├── Summarizer (TextRank / LexRank / BART) -> Node Summaries
                 ├── Topic Modeling (LDA Clustering)       -> Concept Clusters
                 ├── Visualization (WordCloud / Co-occurrence Network)
                 └── Evaluation (Graph Topology, C_v Coherence, ROUGE Metrics)
                """,
                language="text",
            )


if __name__ == "__main__":
    main()
