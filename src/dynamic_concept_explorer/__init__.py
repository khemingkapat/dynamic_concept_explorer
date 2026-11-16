"""
Dynamic Concept Explorer package.
Baseline modules extracted directly from text_analytics coursework.
"""

from . import evaluation, scraper, summarizer, topic_modeling, visualization

__all__ = ["evaluation", "scraper", "summarizer", "topic_modeling", "visualization"]

def main():
    print("Dynamic Concept Explorer initialized.")
