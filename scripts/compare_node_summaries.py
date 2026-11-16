"""Compare the README summarization methods on live Wikipedia articles.

Usage:
    uv run python scripts/compare_node_summaries.py [--bart] [Article_Title ...]

For each topic it prints the 2-sentence node overview from:

* TextRank over the lead section (the default used for nodes),
* TextRank over the full article,
* LexRank over the full article,
* BART over the lead section (only with ``--bart``; downloads the model on first use),

with latency, plus ROUGE-1/2 F1 of the full-article methods against the article lead (the
human-written abstract). Pages are fetched through the scraper cache, so reruns are offline.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter

from dynamic_concept_explorer import scraper, summarizer

DEFAULT_TOPICS = [
    "Machine_learning",
    "Photosynthesis",
    "French_Revolution",
    "General_relativity",
    "Game_theory",
    "Natural_language_processing",
]


def _ngrams(text: str, n: int) -> Counter:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(candidate: str, reference: str, n: int) -> float:
    cand, ref = _ngrams(candidate, n), _ngrams(reference, n)
    overlap = sum((cand & ref).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(cand.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def main(topics: list[str], use_bart: bool) -> None:
    variants = [
        ("TextRank lead", {"method": "textrank", "scope": "lead"}),
        ("TextRank full", {"method": "textrank", "scope": "full"}),
        ("LexRank full", {"method": "lexrank", "scope": "full"}),
    ]
    if use_bart:
        variants.append(("BART lead", {"method": "bart", "scope": "lead"}))

    rows = []
    for topic in topics:
        soup = scraper.fetch_wikipedia_page(f"{scraper.WIKIPEDIA_BASE_URL}/wiki/{topic}")
        body = scraper.get_body_content(soup)
        if body is None:
            print(f"!! {topic}: no bodyContent")
            continue
        article = scraper.extract_article_text(body)
        n_sentences = len(summarizer.split_sentences(article.full))

        print(f"\n=== {topic} ({n_sentences} sentences) ===")
        cells = [topic, str(n_sentences)]
        for label, kwargs in variants:
            start = time.perf_counter()
            summary = summarizer.summarize_node(article.lead, article.body, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"[{label} -> {summary.method}/{summary.scope}] {summary.text}")

            cells.append(f"{elapsed_ms:.1f}")
            if kwargs["scope"] == "full":
                cells.append(f"{rouge_n_f1(summary.text, article.lead, 1):.3f}")
                cells.append(f"{rouge_n_f1(summary.text, article.lead, 2):.3f}")
        rows.append(cells)

    header = ["Topic", "Sents", "TR-lead ms", "TR-full ms", "TR-full R-1", "TR-full R-2"]
    header += ["LR-full ms", "LR-full R-1", "LR-full R-2"]
    if use_bart:
        header.append("BART ms")
    print("\n| " + " | ".join(header) + " |")
    print("| :--- |" + " ---: |" * (len(header) - 1))
    for cells in rows:
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if a != "--bart"] or DEFAULT_TOPICS, "--bart" in args)
