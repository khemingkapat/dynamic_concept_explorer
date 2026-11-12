"""
Dynamic Concept Explorer package.
Baseline modules extracted directly from text_analytics coursework.
"""

from . import scraper
from . import summarizer
from . import topic_modeling
from . import visualization

__all__ = ["scraper", "summarizer", "topic_modeling", "visualization"]

def main():
    print("Dynamic Concept Explorer initialized.")
