"""
Verification script for the baseline extracted modules.
Tests scraper, summarizer, topic_modeling, and visualization functions.
"""

from dynamic_concept_explorer import scraper, summarizer, topic_modeling, visualization

def test_imports_and_signatures():
    print("1. Testing Scraper exports...")
    assert hasattr(scraper, "fetch_wikipedia_page")
    assert hasattr(scraper, "get_body_content")
    assert hasattr(scraper, "extract_section_keywords")
    print("   -> Scraper exports OK.")

    print("2. Testing Summarizer exports...")
    assert hasattr(summarizer, "textrank")
    assert hasattr(summarizer, "lexrank_summary")
    assert hasattr(summarizer, "bart_summary")
    # Quick TextRank test
    sample_text = (
        "Natural language processing is an interdisciplinary subfield of computer science and linguistics. "
        "It is primarily concerned with giving computers the ability to support and manipulate human language. "
        "It involves processing natural language datasets, such as text corpora or speech corpora. "
        "Text summarization is an important task in natural language processing."
    )
    ranked = summarizer.textrank(sample_text, ratio=0.5)
    print(f"   -> TextRank test result ({len(ranked)} sentences):", ranked)

    print("3. Testing Topic Modeling exports...")
    assert hasattr(topic_modeling, "preprocess_text")
    tokens = topic_modeling.preprocess_text("Natural language processing explores topic models.")
    print("   -> Preprocessing tokens:", tokens)

    print("4. Testing Visualization exports...")
    assert hasattr(visualization, "generate_section_wordcloud")
    assert hasattr(visualization, "build_cooccurrence_graph")
    G = visualization.build_cooccurrence_graph("graph concept node edge relationship")
    print(f"   -> Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("\nAll baseline modules verified successfully!")

if __name__ == "__main__":
    test_imports_and_signatures()
