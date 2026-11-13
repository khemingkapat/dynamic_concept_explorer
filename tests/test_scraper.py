"""Offline unit tests for the Wikipedia scraper module."""

from __future__ import annotations

import math
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from dynamic_concept_explorer import scraper


def _make_page(sections_html: str) -> BeautifulSoup:
    html = f"<html><body><div id='bodyContent'>{sections_html}</div></body></html>"
    return BeautifulSoup(html, "html.parser")


def test_extract_link_stats_deduplicates_and_skips_boilerplate() -> None:
    soup = _make_page(
        """
        <section data-mw-section-id="0">
            <p>
                <a href="/wiki/Machine_learning#History">Machine learning</a>
                uses <a href="/wiki/Statistics">statistics</a>[1]
                <a href="/wiki/File:Example.png">File</a>
                <a href="/wiki/Main_Page">Main</a>
                <a href="/wiki/Statistics">[1]</a>
            </p>
        </section>
        <section data-mw-section-id="1">
            <h2>Applications</h2>
            <p>
                More on <a href="/wiki/Machine_learning">ML</a> and
                <a href="/wiki/Deep_learning">deep learning</a>.
            </p>
        </section>
        <section data-mw-section-id="2">
            <h2>See also<span class="mw-editsection">[edit]</span></h2>
            <p><a href="/wiki/Ignored_See_Also">Ignored</a></p>
        </section>
        <section data-mw-section-id="3">
            <h2>References</h2>
            <p><a href="/wiki/Ignored_Ref">Ignored Ref</a></p>
        </section>
        """
    )
    body = scraper.get_body_content(soup)
    assert body is not None

    stats = scraper.extract_link_stats(body)
    assert set(stats.keys()) == {
        "/wiki/Machine_learning",
        "/wiki/Statistics",
        "/wiki/Deep_learning",
    }
    assert stats["/wiki/Machine_learning"].count == 2
    assert stats["/wiki/Machine_learning"].in_lead is True
    assert stats["/wiki/Statistics"].count == 1
    assert stats["/wiki/Statistics"].in_lead is True
    assert stats["/wiki/Deep_learning"].count == 1
    assert stats["/wiki/Deep_learning"].in_lead is False


def test_extract_link_stats_handles_absolute_urls_and_nested_sections() -> None:
    soup = _make_page(
        """
        <section data-mw-section-id="1">
            <h2>Parent</h2>
            <p><a href="https://en.wikipedia.org/wiki/Artificial_intelligence#History">AI</a></p>
            <section data-mw-section-id="2">
                <h3>Nested</h3>
                <p>
                    <a href="https://en.wikipedia.org/wiki/Neural_network">NN</a>
                    <a href="https://arxiv.org/abs/1234.5678">External</a>
                </p>
            </section>
        </section>
        """
    )
    body = scraper.get_body_content(soup)
    assert body is not None

    stats = scraper.extract_link_stats(body)
    assert set(stats.keys()) == {"/wiki/Artificial_intelligence", "/wiki/Neural_network"}
    assert stats["/wiki/Neural_network"].count == 1


def test_rank_outbound_links_default_and_custom_scorer() -> None:
    soup = _make_page(
        """
        <section data-mw-section-id="0">
            <p><a href="/wiki/Lead_Only">Lead Only</a></p>
        </section>
        <section data-mw-section-id="1">
            <h2>Details</h2>
            <p>
                <a href="/wiki/Body_Frequent">BF</a>
                <a href="/wiki/Body_Frequent">BF</a>
                <a href="/wiki/Body_Frequent">BF</a>
                <a href="/wiki/Body_Once">BO</a>
            </p>
        </section>
        """
    )
    body = scraper.get_body_content(soup)
    assert body is not None

    ranked = scraper.rank_outbound_links(body, top_k=2)
    assert [item.href for item in ranked] == ["/wiki/Lead_Only", "/wiki/Body_Frequent"]
    assert math.isclose(ranked[0].score, 2.0 + math.log1p(1))
    assert math.isclose(ranked[1].score, math.log1p(3))

    # Dependency-injected scorer that ranks purely by occurrence count
    by_count = scraper.rank_outbound_links(body, top_k=2, scorer=lambda s: float(s.count))
    assert by_count[0].href == "/wiki/Body_Frequent"


def test_discover_neighborhood_in_degree_and_fallback() -> None:
    pages = {
        "https://en.wikipedia.org/wiki/Seed": _make_page(
            """
            <section data-mw-section-id="0">
                <p>
                    <a href="/wiki/Core_A">Core A</a>
                    <a href="/wiki/Core_B">Core B</a>
                    <a href="/wiki/Core_C">Core C</a>
                    <a href="/wiki/Leaf_D">Leaf D</a>
                </p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Core_A": _make_page(
            """
            <section data-mw-section-id="0">
                <p><a href="/wiki/Core_B">B</a> <a href="/wiki/Core_C">C</a></p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Core_B": _make_page(
            """
            <section data-mw-section-id="0">
                <p><a href="/wiki/Core_A">A</a> <a href="/wiki/Core_C">C</a></p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Core_C": _make_page(
            """
            <section data-mw-section-id="0">
                <p><a href="/wiki/Core_A">A</a> <a href="/wiki/Core_B">B</a></p>
            </section>
            """
        ),
        "https://en.wikipedia.org/wiki/Leaf_D": _make_page(
            "<section data-mw-section-id='0'><p>No links</p></section>"
        ),
    }

    # With min_nodes=0, Leaf_D (in_degree=0) is pruned
    strict = scraper.discover_neighborhood(
        "https://en.wikipedia.org/wiki/Seed",
        top_k=10,
        min_in_degree=2,
        min_nodes=0,
        fetcher=lambda url: pages[url],
    )
    assert [c.href for c in strict.candidates] == [
        "/wiki/Core_A",
        "/wiki/Core_B",
        "/wiki/Core_C",
    ]
    assert all(c.in_degree == 2 for c in strict.candidates)
    assert [p.href for p in strict.pruned] == ["/wiki/Leaf_D"]
    assert "/wiki/Leaf_D" not in strict.edges["/wiki/Seed"]

    # With min_nodes=4, fallback tops up with Leaf_D
    with_fallback = scraper.discover_neighborhood(
        "https://en.wikipedia.org/wiki/Seed",
        top_k=10,
        min_in_degree=2,
        min_nodes=4,
        fetcher=lambda url: pages[url],
    )
    assert len(with_fallback.candidates) == 4
    assert with_fallback.pruned == []


def test_discover_neighborhood_handles_fetch_errors() -> None:
    seed_soup = _make_page(
        "<section data-mw-section-id='0'><p><a href='/wiki/Broken'>Broken</a></p></section>"
    )

    def flaky_fetcher(url: str) -> BeautifulSoup:
        if url.endswith("/Seed"):
            return seed_soup
        raise requests.RequestException("network error")

    result = scraper.discover_neighborhood(
        "/wiki/Seed",
        min_in_degree=1,
        min_nodes=1,
        fetcher=flaky_fetcher,
    )
    assert [c.href for c in result.candidates] == ["/wiki/Broken"]
    assert result.edges["/wiki/Broken"] == set()


def test_fetch_wikipedia_page_uses_memory_and_disk_cache(tmp_path: Path, monkeypatch) -> None:
    scraper.clear_page_cache(tmp_path)
    calls = 0

    class DummyResponse:
        text = "<div id='bodyContent'><p>Cached</p></div>"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> DummyResponse:
        nonlocal calls
        calls += 1
        return DummyResponse()

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    url = "https://en.wikipedia.org/wiki/Cache_Test"
    soup1 = scraper.fetch_wikipedia_page(url, cache_dir=tmp_path)
    assert calls == 1
    assert soup1.find("p").get_text() == "Cached"

    # Second call hits memory cache
    scraper.fetch_wikipedia_page(url, cache_dir=tmp_path)
    assert calls == 1

    # Clearing memory cache still hits disk cache
    scraper.clear_page_cache(cache_dir=None)
    soup3 = scraper.fetch_wikipedia_page(url, cache_dir=tmp_path)
    assert calls == 1
    assert soup3.find("p").get_text() == "Cached"

    # Bypassing cache triggers a fresh network call
    scraper.fetch_wikipedia_page(url, cache_dir=tmp_path, use_cache=False)
    assert calls == 2
