"""Compare lead-paragraph extraction against full-body TextRank on live Wikipedia articles.

Usage:
    uv run python scripts/compare_node_summaries.py [Article_Title ...]

For each topic it prints both 2-sentence overviews, their latency, and ROUGE-1/2 F1 of the
body-only TextRank summary against the article lead (the human-written abstract). It also times
the vectorised PageRank against the coursework ``nx.pagerank`` TextRank on the same sentences.
Pages are fetched through the scraper cache, so reruns are offline.
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


def _time_ms(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def main(topics: list[str]) -> None:
    rows = []
    for topic in topics:
        soup = scraper.fetch_wikipedia_page(f"{scraper.WIKIPEDIA_BASE_URL}/wiki/{topic}")
        body = scraper.get_body_content(soup)
        if body is None:
            print(f"!! {topic}: no bodyContent")
            continue
        article = scraper.extract_article_text(body)

        lead = summarizer.summarize_node(article.lead, article.body, method="lead")
        rank = summarizer.summarize_node("", article.body, method="textrank")

        n_sentences = len(summarizer.split_sentences(article.body))
        lead_ms = _time_ms(lambda a=article: summarizer.summarize_node(a.lead, a.body))
        rank_ms = _time_ms(
            lambda a=article: summarizer.summarize_node("", a.body, method="textrank")
        )
        base_ms = _time_ms(lambda a=article: summarizer.textrank(a.body, ratio=0.01), repeats=1)

        rows.append(
            (
                topic,
                n_sentences,
                lead_ms,
                rank_ms,
                base_ms,
                rouge_n_f1(rank.text, article.lead, 1),
                rouge_n_f1(rank.text, article.lead, 2),
            )
        )
        print(f"\n=== {topic} ({n_sentences} body sentences) ===")
        print(f"[lead]     {lead.text}")
        print(f"[textrank] {rank.text}")

    header = ["Topic", "Body sents", "Lead ms", "TextRank ms", "Baseline nx ms", "R-1", "R-2"]
    print("\n| " + " | ".join(header) + " |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for topic, n, lead_ms, rank_ms, base_ms, r1, r2 in rows:
        cells = [topic, str(n), f"{lead_ms:.1f}", f"{rank_ms:.1f}", f"{base_ms:.1f}"]
        cells += [f"{r1:.3f}", f"{r2:.3f}"]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_TOPICS)
