"""Wikipedia scraping, caching, outbound link ranking, and 2-pass neighborhood pruning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
DEFAULT_HEADERS = {"User-Agent": "DynamicConceptExplorer/1.0"}
DEFAULT_CACHE_DIR = Path(".cache/wiki")
DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

SKIP_PATTERN = re.compile(r"^(\[\d+\]|\s*)$")
DEFAULT_SKIP_SECTIONS = frozenset(
    {
        "see also",
        "references",
        "external links",
        "notes",
        "further reading",
        "bibliography",
        "sources",
    }
)
EXCLUDED_HREFS = frozenset({"/wiki/Main_Page"})

_MEMORY_CACHE: dict[str, tuple[float, str]] = {}


@dataclass(frozen=True)
class LinkStats:
    """Aggregated statistics for a canonical Wikipedia link within an article."""

    href: str
    text: str
    count: int
    in_lead: bool


@dataclass(frozen=True)
class RankedLink:
    """Scored candidate link with optional neighborhood in-degree from 2-pass filtering."""

    href: str
    text: str
    count: int
    in_lead: bool
    score: float
    in_degree: int = 0


@dataclass(frozen=True)
class NeighborhoodResult:
    """Result of 2-pass neighborhood discovery and in-degree pruning."""

    seed_url: str
    candidates: list[RankedLink]
    pruned: list[RankedLink]
    edges: dict[str, set[str]]


LinkScorer = Callable[[LinkStats], float]
PageFetcher = Callable[[str], BeautifulSoup]


def default_link_scorer(stats: LinkStats) -> float:
    """Score a candidate link by lead-section prominence and log occurrence frequency."""
    lead_bonus = 2.0 if stats.in_lead else 0.0
    return lead_bonus + math.log1p(stats.count)


def _is_fresh(fetched_at: float, ttl_seconds: float | None) -> bool:
    if ttl_seconds is None:
        return True
    return (time.time() - fetched_at) <= ttl_seconds


def _cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_cached_html(
    url: str,
    cache_dir: Path,
    ttl_seconds: float | None,
) -> str | None:
    if url in _MEMORY_CACHE:
        fetched_at, html = _MEMORY_CACHE[url]
        if _is_fresh(fetched_at, ttl_seconds):
            return html
        del _MEMORY_CACHE[url]

    path = _cache_path(url, cache_dir)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = float(payload["fetched_at"])
        html = str(payload["html"])
    except (OSError, ValueError, KeyError, TypeError):
        return None

    if not _is_fresh(fetched_at, ttl_seconds):
        return None

    _MEMORY_CACHE[url] = (fetched_at, html)
    return html


def _write_cached_html(url: str, html: str, cache_dir: Path) -> None:
    fetched_at = time.time()
    _MEMORY_CACHE[url] = (fetched_at, html)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"url": url, "fetched_at": fetched_at, "html": html}
    _cache_path(url, cache_dir).write_text(json.dumps(payload), encoding="utf-8")


def clear_page_cache(cache_dir: Path | None = DEFAULT_CACHE_DIR) -> None:
    """Clear the in-memory cache and optionally remove cached files from disk."""
    _MEMORY_CACHE.clear()
    if cache_dir is not None and cache_dir.exists():
        for file_path in cache_dir.glob("*.json"):
            file_path.unlink(missing_ok=True)


def fetch_wikipedia_page(
    url: str,
    headers: dict[str, str] | None = None,
    *,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ttl_seconds: float | None = DEFAULT_CACHE_TTL_SECONDS,
    timeout: float = 10.0,
) -> BeautifulSoup:
    """Fetch a Wikipedia page (with memory + disk caching) and return parsed BeautifulSoup."""
    if use_cache:
        cached_html = _read_cached_html(url, cache_dir, ttl_seconds)
        if cached_html is not None:
            return BeautifulSoup(cached_html, "html.parser")

    response = requests.get(
        url,
        headers=headers or DEFAULT_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    if use_cache:
        _write_cached_html(url, response.text, cache_dir)

    return BeautifulSoup(response.text, "html.parser")


def get_body_content(soup: BeautifulSoup) -> Tag | None:
    """Extract the main body content (`div#bodyContent`) from a parsed Wikipedia page."""
    body = soup.find("div", id="bodyContent")
    return body if isinstance(body, Tag) else None


def _normalize_section_title(section: Tag) -> str:
    heading = section.find(["h1", "h2", "h3", "h4"])
    if heading is not None:
        title = heading.get_text(" ", strip=True)
        return re.sub(r"\s*\[edit\]$", "", title, flags=re.IGNORECASE).strip()

    lines = [line.strip() for line in section.get_text("\n").split("\n") if line.strip()]
    return lines[0] if lines else ""


def _is_lead_section(section: Tag, index: int) -> bool:
    section_id = section.get("data-mw-section-id")
    if section_id is not None:
        return str(section_id) == "0"
    return index == 0 and section.find(["h2", "h3"]) is None


def _canonical_wiki_href(raw_href: str) -> str | None:
    if raw_href.startswith(("http://", "https://")):
        parsed = urlparse(raw_href)
        if not parsed.netloc.endswith("wikipedia.org") or not parsed.path.startswith("/wiki/"):
            return None
        canonical = parsed.path
    elif raw_href.startswith("/wiki/"):
        canonical = raw_href.split("#", 1)[0].split("?", 1)[0]
    else:
        return None

    if ":" in canonical or canonical in EXCLUDED_HREFS or len(canonical) <= len("/wiki/"):
        return None
    return canonical


def _top_level_sections(body: Tag) -> list[Tag]:
    return [sec for sec in body.find_all("section") if sec.find_parent("section") is None]


def extract_section_keywords(body: Tag) -> dict[str, list[str]]:
    """Extract section titles and cleaned outbound hyperlink texts from bodyContent."""
    section_keywords: dict[str, list[str]] = {}
    for section in body.find_all("section"):
        lines = [line.strip() for line in section.text.split("\n") if line.strip()]
        if not lines:
            continue

        key = lines[0]
        section_keywords[key] = [
            hpl.text.strip()
            for hpl in section.find_all("a")
            if not SKIP_PATTERN.match(hpl.text.strip())
        ]
    return section_keywords


def extract_section_links(body: Tag) -> dict[str, list[dict[str, str]]]:
    """Extract section titles and cleaned outbound Wikipedia links with canonical hrefs."""
    section_links: dict[str, list[dict[str, str]]] = {}
    for section in body.find_all("section"):
        lines = [line.strip() for line in section.text.split("\n") if line.strip()]
        if not lines:
            continue

        key = lines[0]
        links: list[dict[str, str]] = []
        for hpl in section.find_all("a"):
            text = hpl.text.strip()
            canonical = _canonical_wiki_href(str(hpl.get("href", "")))
            if not SKIP_PATTERN.match(text) and canonical is not None:
                links.append({"text": text, "href": canonical})
        section_links[key] = links
    return section_links


def extract_link_stats(
    body: Tag,
    *,
    skip_sections: frozenset[str] = DEFAULT_SKIP_SECTIONS,
) -> dict[str, LinkStats]:
    """Aggregate canonical outbound Wikipedia link statistics across non-boilerplate sections."""
    stats_by_href: dict[str, dict[str, object]] = {}
    sections = _top_level_sections(body)
    containers: list[tuple[Tag, bool]] = (
        [(sec, _is_lead_section(sec, idx)) for idx, sec in enumerate(sections)]
        if sections
        else [(body, True)]
    )

    for section, in_lead in containers:
        if not in_lead:
            title = _normalize_section_title(section).lower()
            if title in skip_sections:
                continue

        for anchor in section.find_all("a"):
            text = anchor.get_text(strip=True)
            if SKIP_PATTERN.match(text):
                continue

            canonical = _canonical_wiki_href(str(anchor.get("href", "")))
            if canonical is None:
                continue

            entry = stats_by_href.get(canonical)
            if entry is None:
                stats_by_href[canonical] = {
                    "text": text,
                    "count": 1,
                    "in_lead": in_lead,
                }
            else:
                entry["count"] = int(entry["count"]) + 1
                entry["in_lead"] = bool(entry["in_lead"]) or in_lead

    return {
        href: LinkStats(
            href=href,
            text=str(data["text"]),
            count=int(data["count"]),
            in_lead=bool(data["in_lead"]),
        )
        for href, data in stats_by_href.items()
    }


def rank_outbound_links(
    body: Tag,
    *,
    top_k: int = 20,
    scorer: LinkScorer = default_link_scorer,
    skip_sections: frozenset[str] = DEFAULT_SKIP_SECTIONS,
) -> list[RankedLink]:
    """Rank outbound Wikipedia links using an injectable scorer and return the top-k candidates."""
    if top_k <= 0:
        return []

    stats_map = extract_link_stats(body, skip_sections=skip_sections)
    ranked = [
        RankedLink(
            href=stats.href,
            text=stats.text,
            count=stats.count,
            in_lead=stats.in_lead,
            score=float(scorer(stats)),
        )
        for stats in stats_map.values()
    ]
    ranked.sort(key=lambda item: (-item.score, -item.count, item.href))
    return ranked[:top_k]


def _resolve_seed_url(seed_url: str, base_url: str) -> str:
    if seed_url.startswith("/wiki/"):
        return urljoin(base_url, seed_url)
    return seed_url


def _extract_seed_href(seed_url: str) -> str | None:
    marker = "/wiki/"
    idx = seed_url.find(marker)
    if idx == -1:
        return None
    return _canonical_wiki_href(seed_url[idx:])


def discover_neighborhood(
    seed_url: str,
    *,
    top_k: int = 20,
    min_in_degree: int = 2,
    min_nodes: int = 5,
    scorer: LinkScorer = default_link_scorer,
    skip_sections: frozenset[str] = DEFAULT_SKIP_SECTIONS,
    base_url: str = WIKIPEDIA_BASE_URL,
    fetcher: PageFetcher | None = None,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ttl_seconds: float | None = DEFAULT_CACHE_TTL_SECONDS,
) -> NeighborhoodResult:
    """Run 2-pass candidate ranking and neighborhood in-degree pruning from a seed page."""
    resolved_seed = _resolve_seed_url(seed_url, base_url)
    active_fetcher = fetcher or (
        lambda url: fetch_wikipedia_page(
            url,
            use_cache=use_cache,
            cache_dir=cache_dir,
            ttl_seconds=ttl_seconds,
        )
    )

    seed_soup = active_fetcher(resolved_seed)
    seed_body = get_body_content(seed_soup)
    if seed_body is None:
        return NeighborhoodResult(seed_url=resolved_seed, candidates=[], pruned=[], edges={})

    pass1_candidates = rank_outbound_links(
        seed_body,
        top_k=top_k,
        scorer=scorer,
        skip_sections=skip_sections,
    )
    seed_href = _extract_seed_href(resolved_seed)
    if seed_href is not None:
        pass1_candidates = [c for c in pass1_candidates if c.href != seed_href]

    candidate_hrefs = {c.href for c in pass1_candidates}
    in_degree_counts: dict[str, int] = {c.href: 0 for c in pass1_candidates}
    edges: dict[str, set[str]] = {}
    if seed_href is not None:
        edges[seed_href] = set(candidate_hrefs)

    for candidate in pass1_candidates:
        target_url = urljoin(base_url, candidate.href)
        try:
            soup = active_fetcher(target_url)
        except requests.RequestException:
            edges[candidate.href] = set()
            continue

        body = get_body_content(soup)
        if body is None:
            edges[candidate.href] = set()
            continue

        outbound = extract_link_stats(body, skip_sections=skip_sections)
        neighbor_links = (set(outbound.keys()) & candidate_hrefs) - {candidate.href}
        edges[candidate.href] = neighbor_links
        for target_href in neighbor_links:
            in_degree_counts[target_href] += 1

    scored_with_degree = [
        RankedLink(
            href=c.href,
            text=c.text,
            count=c.count,
            in_lead=c.in_lead,
            score=c.score,
            in_degree=in_degree_counts[c.href],
        )
        for c in pass1_candidates
    ]

    kept = [c for c in scored_with_degree if c.in_degree >= min_in_degree]
    below_threshold = [c for c in scored_with_degree if c.in_degree < min_in_degree]

    if len(kept) < min_nodes and below_threshold:
        below_threshold.sort(key=lambda item: (-item.in_degree, -item.score, item.href))
        needed = min_nodes - len(kept)
        kept.extend(below_threshold[:needed])
        pruned = below_threshold[needed:]
    else:
        pruned = below_threshold

    kept.sort(key=lambda item: (-item.in_degree, -item.score, item.href))
    kept_hrefs = {c.href for c in kept}
    filtered_edges = {
        src: targets & kept_hrefs
        for src, targets in edges.items()
        if src == seed_href or src in kept_hrefs
    }

    return NeighborhoodResult(
        seed_url=resolved_seed,
        candidates=kept,
        pruned=pruned,
        edges=filtered_edges,
    )
