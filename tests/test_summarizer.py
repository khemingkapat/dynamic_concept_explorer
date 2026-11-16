"""Offline unit tests for article text extraction and node summarization."""

from __future__ import annotations

import numpy as np
import pytest
from bs4 import BeautifulSoup

from dynamic_concept_explorer import scraper, summarizer

# Short academic articles: (lead, body). Leads open with a definition, as Wikipedia leads do.
ACADEMIC_TOPICS = {
    "machine_learning": (
        "Machine learning is a field of study in artificial intelligence concerned with "
        "statistical algorithms that learn from data. It allows computers to perform tasks "
        "without being explicitly programmed for each one. Deep learning is a notable subfield.",
        "The term machine learning was coined in 1959 by Arthur Samuel while working on "
        "computer checkers. Supervised learning algorithms build a model from labelled "
        "training data. Unsupervised learning algorithms find structure in unlabelled data. "
        "Reinforcement learning trains agents by rewarding desired behaviour in an environment.",
    ),
    "photosynthesis": (
        "Photosynthesis is a biological process that converts light energy into chemical "
        "energy stored in sugars. Plants, algae, and cyanobacteria perform photosynthesis "
        "and release oxygen as a byproduct of the reaction.",
        "The light-dependent reactions take place in the thylakoid membranes of chloroplasts. "
        "Chlorophyll absorbs mostly blue and red light and reflects green light. "
        "The Calvin cycle uses energy from light reactions to fix carbon dioxide into sugars.",
    ),
    "french_revolution": (
        "The French Revolution was a period of political and societal change in France that "
        "began in 1789. It ended with the formation of the French Consulate in November 1799 "
        "and reshaped ideas about citizenship across Europe.",
        "Financial crisis and widespread social distress led to the convocation of the "
        "Estates General in May 1789. The storming of the Bastille on 14 July became a symbol "
        "of the revolution. The monarchy was abolished and a republic was proclaimed in 1792.",
    ),
    "general_relativity": (
        "General relativity is the geometric theory of gravitation published by Albert "
        "Einstein in 1915. It describes gravity as a property of the curvature of spacetime "
        "caused by the presence of mass and energy.",
        "The Einstein field equations relate spacetime curvature to the energy and momentum "
        "of matter. The theory predicts gravitational time dilation and the bending of light "
        "by massive objects. Gravitational waves were first directly observed in 2015.",
    ),
}


def _body(html: str):
    soup = BeautifulSoup(
        f"<html><body><div id='bodyContent'>{html}</div></body></html>", "html.parser"
    )
    body = scraper.get_body_content(soup)
    assert body is not None
    return body


def test_extract_article_text_splits_lead_and_strips_noise() -> None:
    body = _body(
        """
        <p class="mw-empty-elt"> </p>
        <table class="infobox"><tr><td><p>Infobox prose is not article text.</p></td></tr></table>
        <p><b>Graph theory</b> (<b>GT</b>) studies graphs<sup class="reference">[1]</sup>
           made of <a href="/wiki/Vertex">vertices</a>.
           <sup class="noprint">[citation needed]</sup></p>
        <div class="mw-heading mw-heading2">
            <h2>History<span class="mw-editsection">[edit]</span></h2>
        </div>
        <p>Euler solved the bridges problem in 1736.[2] Energy is
           <span class="mwe-math-element"><math>E=mc^2</math></span>conserved.</p>
        <h2>See also</h2>
        <p>Ignored see-also prose.</p>
        <h2>References</h2>
        <p>Ignored reference prose.</p>
        """
    )

    article = scraper.extract_article_text(body)

    assert article.lead == "Graph theory (GT) studies graphs made of vertices."
    assert article.body == "Euler solved the bridges problem in 1736. Energy is conserved."
    assert article.full == f"{article.lead}\n\n{article.body}"
    # Extraction must not mutate the parsed page (link ranking reuses the same soup).
    assert body.find("sup", class_="reference") is not None


def test_extract_article_text_handles_section_wrapped_markup() -> None:
    body = _body(
        """
        <section data-mw-section-id="0">
            <p>Lead paragraph one.</p><p>Lead paragraph two.</p>
        </section>
        <section data-mw-section-id="1"><h2>Overview</h2><p>Body paragraph.</p>
            <section data-mw-section-id="2"><h3>Detail</h3><p>Nested paragraph.</p></section>
        </section>
        <section data-mw-section-id="3"><h2>External links</h2><p>Ignored.</p></section>
        """
    )

    article = scraper.extract_article_text(body)

    assert article.lead == "Lead paragraph one.\n\nLead paragraph two."
    assert article.body == "Body paragraph.\n\nNested paragraph."


def test_rank_sentences_agrees_with_baseline_textrank() -> None:
    lead, body = ACADEMIC_TOPICS["machine_learning"]
    text = f"{lead} {body}"
    sentences = summarizer.split_sentences(text)

    scores = summarizer.rank_sentences(sentences)
    best = sentences[int(np.argmax(scores))]

    assert scores.sum() == pytest.approx(1.0, abs=1e-6)
    # The coursework TextRank must pick the same most-central sentence.
    assert summarizer.textrank(text, ratio=1 / len(sentences)) == [best]
    assert best in summarizer.textrank_sentences(text, n_sentences=2)


@pytest.mark.parametrize("topic", sorted(ACADEMIC_TOPICS))
def test_summarize_node_default_is_textrank_over_lead_section(topic: str) -> None:
    lead, body = ACADEMIC_TOPICS[topic]
    lead_source = summarizer.split_sentences(lead)

    summary = summarizer.summarize_node(lead, body)

    assert (summary.method, summary.scope) == ("textrank", "lead")
    assert len(summary.sentences) == 2
    positions = [lead_source.index(sentence) for sentence in summary.sentences]
    assert positions == sorted(positions)
    assert summary.text == " ".join(summary.sentences)
    assert summary.elapsed_ms >= 0.0


@pytest.mark.parametrize("method", ["textrank", "lexrank"])
@pytest.mark.parametrize("topic", sorted(ACADEMIC_TOPICS))
def test_summarize_node_full_scope_is_extractive_and_ordered(topic: str, method: str) -> None:
    lead, body = ACADEMIC_TOPICS[topic]
    source = summarizer.split_sentences(f"{lead}\n\n{body}")

    summary = summarizer.summarize_node(lead, body, method=method, scope="full")

    assert (summary.method, summary.scope) == (method, "full")
    assert len(summary.sentences) == 2
    positions = [source.index(sentence) for sentence in summary.sentences]
    assert positions == sorted(positions)


def test_summarize_node_lexrank_accepts_background_corpus() -> None:
    lead, body = ACADEMIC_TOPICS["photosynthesis"]
    background = [
        f"{a} {b}" for name, (a, b) in ACADEMIC_TOPICS.items() if name != "photosynthesis"
    ]

    summary = summarizer.summarize_node(lead, body, method="lexrank", background=background)

    assert (summary.method, summary.scope) == ("lexrank", "lead")
    assert len(summary.sentences) == 2
    assert all(sentence in lead for sentence in summary.sentences)


def test_summarize_node_uses_full_article_when_lead_is_thin() -> None:
    _, body = ACADEMIC_TOPICS["photosynthesis"]

    thin_lead = summarizer.summarize_node("Photosynthesis.", body)
    no_text = summarizer.summarize_node("", "")

    assert (thin_lead.method, thin_lead.scope) == ("textrank", "full")
    assert len(thin_lead.sentences) == 2
    assert all(sentence in body for sentence in thin_lead.sentences)
    assert no_text.text == ""
    assert no_text.sentences == []


def test_summarize_node_skips_fragments_and_respects_n_sentences() -> None:
    lead = (
        "Main article below. Topology is the branch of mathematics concerned with "
        "properties preserved under continuous deformation. It studies stretching and "
        "bending but not tearing. Knot theory is one of its best known subfields today."
    )

    one = summarizer.summarize_node(lead, n_sentences=1)
    three = summarizer.summarize_node(lead, n_sentences=3)

    assert len(one.sentences) == 1
    assert len(three.sentences) == 3
    assert three.sentences[0].startswith("Topology is the branch")
    assert not any("Main article" in sentence for sentence in three.sentences)


def test_summarize_node_bart_on_demand_and_fallback() -> None:
    lead, body = ACADEMIC_TOPICS["general_relativity"]
    seen: list[str] = []

    def fake_bart(text: str) -> str:
        seen.append(text)
        return "Gravity is curved spacetime. Einstein published it in 1915."

    def broken_bart(text: str) -> str:
        raise OSError("model weights are not available offline")

    generated = summarizer.summarize_node(lead, body, method="bart", abstractive=fake_bart)
    degraded = summarizer.summarize_node(lead, body, method="bart", abstractive=broken_bart)

    assert seen == [lead]  # only the lead is sent to the model, so it stays under 1024 tokens
    assert generated.method == "bart"
    assert generated.text == "Gravity is curved spacetime. Einstein published it in 1915."
    assert degraded.method == "textrank (bart unavailable)"
    assert degraded.sentences == summarizer.summarize_node(lead, body).sentences


def test_summarize_node_bart_keeps_only_complete_sentences_up_to_limit() -> None:
    lead, body = ACADEMIC_TOPICS["photosynthesis"]
    cut_off = "Plants convert light into sugar. Oxygen is released. It maintains the Earth's"
    too_long = "First point here. Second point here. Third point here. Fourth point here."

    trimmed = summarizer.summarize_node(lead, body, method="bart", abstractive=lambda _: cut_off)
    capped = summarizer.summarize_node(lead, body, method="bart", abstractive=lambda _: too_long)
    one = summarizer.summarize_node(
        lead, body, method="bart", n_sentences=1, abstractive=lambda _: cut_off
    )
    only_partial = summarizer.summarize_node(
        lead, body, method="bart", abstractive=lambda _: "Photosynthesis is the process by"
    )

    assert trimmed.text == "Plants convert light into sugar. Oxygen is released."
    assert capped.text == "First point here. Second point here."
    assert one.text == "Plants convert light into sugar."
    # A lone cut-off sentence is kept: a partial overview beats an empty node.
    assert (only_partial.method, only_partial.text) == ("bart", "Photosynthesis is the process by")


def test_summarize_node_rejects_unknown_method_and_scope() -> None:
    with pytest.raises(ValueError, match="Unknown summarization method"):
        summarizer.summarize_node("Some lead text.", method="lead")
    with pytest.raises(ValueError, match="Unknown summarization scope"):
        summarizer.summarize_node("Some lead text.", scope="body")
