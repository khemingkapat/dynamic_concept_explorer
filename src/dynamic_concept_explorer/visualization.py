"""
Baseline visualization module extracted directly from:
- web_scraping/Lab3_wiki_activity.ipynb (WordCloud generation per section)
- text_vis/supplement_vis.ipynb (NetworkX graph co-occurrence diagram)
"""

from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from wordcloud import WordCloud

try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass
from nltk.tokenize import word_tokenize


def generate_section_wordcloud(keywords: list[str], section_title: str = "", show: bool = True):
    """
    Exact WordCloud generation matching Cell 8 from Lab3_wiki_activity.ipynb:
    width=800, height=400, background_color="white", collocations=False.
    """
    if not keywords:
        return None

    # Join the keywords into a single string separated by spaces
    text_data = " ".join(keywords)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False,  # Set to False to prevent repeating word pairs
    ).generate(text_data)

    # Display the plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if section_title:
        plt.title(section_title, fontsize=16, fontweight="bold")
    plt.tight_layout(pad=0)
    if show:
        plt.show()
        
    return wordcloud


def build_cooccurrence_graph(text: str) -> nx.Graph:
    """
    Exact graph building logic matching Cell 65 from text_vis/supplement_vis.ipynb:
    Tokenizes words, finds pairwise combinations, sets edge weights.
    """
    words = word_tokenize(text)
    G = nx.Graph()

    for word1, word2 in combinations(words, 2):
        if G.has_edge(word1, word2):
            G[word1][word2]['weight'] += 1
        else:
            G.add_edge(word1, word2, weight=1)

    return G


def draw_network_diagram(G: nx.Graph, title: str = "Text Network Diagram based on Word Co-occurrence", show: bool = True):
    """
    Exact drawing parameters matching Cell 65 from text_vis/supplement_vis.ipynb:
    Node size scaled by degree (deg * 20), spring layout seed 42, skyblue nodes.
    """
    node_size = [deg * 20 for deg in dict(G.degree()).values()]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, 
        pos, 
        with_labels=True, 
        font_size=8, 
        node_size=node_size, 
        font_color='black', 
        edge_color='gray', 
        font_weight='bold', 
        alpha=0.7, 
        node_color='skyblue'
    )
    plt.title(title)
    if show:
        plt.show()
