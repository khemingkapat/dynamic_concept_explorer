"""
Baseline web scraping module extracted directly from:
- web_scraping/Lab3_wiki_activity.ipynb
- web_scraping/Lab3_scraper.ipynb
"""

import re
import requests
from bs4 import BeautifulSoup

# Exact User-Agent and headers from Lab3_wiki_activity.ipynb
DEFAULT_HEADERS = {
    "User-Agent": "MeTryingToGetPassTheRule/1.0"
}

# Exact regex pattern used to skip citations (e.g. [1], [2]) and whitespace
SKIP_PATTERN = re.compile(r"^(\[\d+\]|\s*)$")


def fetch_wikipedia_page(url: str, headers: dict | None = None) -> BeautifulSoup:
    """
    Scrape a Wikipedia page and return the BeautifulSoup parsed object.
    Matches Cell 3 from Lab3_wiki_activity.ipynb.
    """
    if headers is None:
        headers = DEFAULT_HEADERS
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    return soup


def get_body_content(soup: BeautifulSoup):
    """
    Extract the main body content from the parsed Wikipedia soup.
    Matches Cell 4 from Lab3_wiki_activity.ipynb.
    """
    return soup.find("div", id="bodyContent")


def extract_section_keywords(body) -> dict[str, list[str]]:
    """
    Extract section titles and cleaned outbound hyperlink texts from bodyContent.
    Matches Cell 5 & 6 from Lab3_wiki_activity.ipynb exactly.
    """
    section_keywords = dict()
    sections = body.find_all("section")
    for section in sections:
        lines = [line.strip() for line in section.text.split("\n") if line.strip()]
        if not lines:
            continue
        
        key = lines[0]
        
        cleaned_hpls = [
            hpl.text.strip()
            for hpl in section.find_all("a")
            if not SKIP_PATTERN.match(hpl.text.strip())
        ]
        
        section_keywords[key] = cleaned_hpls
        
    return section_keywords


def extract_section_links(body) -> dict[str, list[dict]]:
    """
    Helper extending the baseline to also retain outbound hrefs 
    along with anchor text for network graph building.
    """
    section_links = dict()
    sections = body.find_all("section")
    for section in sections:
        lines = [line.strip() for line in section.text.split("\n") if line.strip()]
        if not lines:
            continue
        
        key = lines[0]
        
        links = []
        for hpl in section.find_all("a"):
            text = hpl.text.strip()
            href = hpl.get("href", "")
            if not SKIP_PATTERN.match(text) and href.startswith("/wiki/") and not ":" in href:
                links.append({"text": text, "href": href})
        
        section_links[key] = links
        
    return section_links
