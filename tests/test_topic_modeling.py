"""Offline unit tests for unsupervised LDA topic modeling and scraper text extraction."""

from __future__ import annotations

import math

import pytest
from bs4 import BeautifulSoup

from dynamic_concept_explorer import scraper, topic_modeling


def _make_wiki_page(sections_html: str) -> BeautifulSoup:
    html = f"<html><body><div id='bodyContent'>{sections_html}</div></body></html>"
    return BeautifulSoup(html, "html.parser")


def test_preprocess_text_normalizes_lemmatizes_and_extra_stopwords() -> None:
    text = "Café neural networks were running models and learning algorithms."
    tokens = topic_modeling.preprocess_text(text)
    assert "cafe" in tokens
    assert "network" in tokens
    assert "run" in tokens
    assert "model" in tokens
    assert "were" not in tokens

    filtered = topic_modeling.preprocess_text(
        text,
        extra_stopwords={"neural", "Learning"},
    )
    assert "neural" not in filtered
    assert "learn" not in filtered


def test_build_corpus_and_dictionary_respects_hyperparams() -> None:
    docs = [
        ["alpha", "beta", "common"],
        ["alpha", "gamma", "common"],
        ["delta", "epsilon", "common"],
    ]
    dictionary, corpus = topic_modeling.build_corpus_and_dictionary(
        docs,
        no_below=2,
        no_above=0.7,
    )
    vocab = set(dictionary.token2id.keys())
    # "alpha" in 2/3 (66.7%) -> kept; "common" in 3/3 (100%) -> dropped; rare -> dropped
    assert vocab == {"alpha"}
    assert len(corpus) == 3


def test_cluster_topic_nodes_assigns_topics_and_descriptors() -> None:
    documents = {
        "/wiki/Neural_network": (
            "Neural networks use layers of artificial neurons, backpropagation, "
            "activation functions, and gradient descent for deep learning."
        ),
        "/wiki/Convolutional_neural_network": (
            "Convolutional neural networks apply learned filters and pooling layers "
            "for image recognition and deep learning vision tasks."
        ),
        "/wiki/Backpropagation": (
            "Backpropagation computes gradients through neural network layers "
            "to update weights during gradient descent optimization."
        ),
        "/wiki/Bayesian_inference": (
            "Bayesian inference updates prior probability distributions using likelihood "
            "and Bayes theorem to compute posterior probability."
        ),
        "/wiki/Markov_chain": (
            "A Markov chain models stochastic probability transitions between states "
            "using conditional probability distributions."
        ),
        "/wiki/Probability_distribution": (
            "Probability distributions assign likelihoods to random variables, "
            "including prior and posterior Bayesian models."
        ),
        "/wiki/Graph_theory": (
            "Graph theory studies vertices, edges, adjacency matrices, shortest paths, "
            "and spanning trees in discrete networks."
        ),
        "/wiki/Shortest_path_problem": (
            "Shortest path algorithms traverse weighted graph vertices and edges "
            "using Dijkstra priority queues and spanning trees."
        ),
        "/wiki/Spanning_tree": (
            "A spanning tree connects all vertices of an undirected graph "
            "with minimal edge weight and no cycles."
        ),
    }

    result = topic_modeling.cluster_topic_nodes(
        documents,
        num_topics=3,
        no_below=2,
        no_above=0.85,
        alpha=0.1,
        eta=0.1,
        num_keywords=4,
        random_state=42,
    )

    assert result.num_topics == 3
    assert set(result.topics.keys()) == {0, 1, 2}
    assert set(result.assignments.keys()) == set(documents.keys())

    for topic_id, desc in result.topics.items():
        assert desc.topic_id == topic_id
        assert desc.label
        assert 1 <= len(desc.keywords) <= 4

    for node_id, assignment in result.assignments.items():
        assert assignment.node_id == node_id
        assert 0 <= assignment.topic_id < 3
        assert 0.0 < assignment.confidence <= 1.0
        assert len(assignment.distribution) == 3
        assert math.isclose(sum(prob for _, prob in assignment.distribution), 1.0, rel_tol=1e-4)

    # Verify coherent clustering: same-domain articles share dominant topics
    assert (
        result.assignments["/wiki/Neural_network"].topic_id
        == result.assignments["/wiki/Convolutional_neural_network"].topic_id
    )
    assert (
        result.assignments["/wiki/Graph_theory"].topic_id
        == result.assignments["/wiki/Spanning_tree"].topic_id
    )


def test_cluster_topic_nodes_validates_and_handles_edge_cases() -> None:
    with pytest.raises(ValueError, match="empty"):
        topic_modeling.cluster_topic_nodes({})

    with pytest.raises(ValueError, match="num_topics"):
        topic_modeling.cluster_topic_nodes({"/wiki/A": "alpha beta"}, num_topics=0)

    with pytest.raises(ValueError, match="vocabulary"):
        topic_modeling.cluster_topic_nodes({"/wiki/A": "the and is of"})

    # Empty-BoW document alongside valid docs should fall back to uniform distribution
    docs = {
        "/wiki/Valid_1": "neural network gradient descent optimization layer",
        "/wiki/Valid_2": "bayesian probability prior posterior distribution",
        "/wiki/Empty_Doc": "the and is",
    }
    result = topic_modeling.cluster_topic_nodes(docs, num_topics=4, no_below=99)
    assert result.num_topics == 3  # bounded by len(docs)
    empty_assign = result.assignments["/wiki/Empty_Doc"]
    assert empty_assign.topic_id == 0
    assert empty_assign.confidence == 0.0
    assert len(empty_assign.distribution) == 3


def test_fetch_neighborhood_documents_extracts_candidates_and_canonical_seed() -> None:
    pages = {
        "https://en.wikipedia.org/wiki/Seed_Topic": _make_wiki_page(
            """
            <section data-mw-section-id="0">
                <p>Seed topic lead paragraph about machine learning.</p>
            </section>
            <section data-mw-section-id="1">
                <h2>References</h2>
                <p>Boilerplate citation text that must be ignored.</p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Active_Candidate": _make_wiki_page(
            """
            <section data-mw-section-id="0">
                <p>Active candidate article on neural networks.</p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Pruned_Node": _make_wiki_page(
            """
            <section data-mw-section-id="0">
                <p>Pruned node text that should not be fetched.</p>
            </section>
            """
        ),
    }
    fetched_urls: list[str] = []

    def mock_fetcher(url: str) -> BeautifulSoup:
        fetched_urls.append(url)
        return pages[url]

    neighborhood = scraper.NeighborhoodResult(
        seed_url="https://en.wikipedia.org/wiki/Seed_Topic",
        candidates=[
            scraper.RankedLink(
                href="/wiki/Active_Candidate",
                text="Active Candidate",
                count=3,
                in_lead=True,
                score=3.0,
                in_degree=2,
            )
        ],
        pruned=[
            scraper.RankedLink(
                href="/wiki/Pruned_Node",
                text="Pruned Node",
                count=1,
                in_lead=False,
                score=0.7,
                in_degree=0,
            )
        ],
        edges={"/wiki/Seed_Topic": {"/wiki/Active_Candidate"}},
    )

    docs = scraper.fetch_neighborhood_documents(neighborhood, fetcher=mock_fetcher)
    assert set(docs.keys()) == {"/wiki/Seed_Topic", "/wiki/Active_Candidate"}
    assert "Boilerplate citation text" not in docs["/wiki/Seed_Topic"]
    assert "machine learning" in docs["/wiki/Seed_Topic"]
    assert "https://en.wikipedia.org/wiki/Pruned_Node" not in fetched_urls
