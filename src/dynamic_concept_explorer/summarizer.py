"""
Baseline summarization module extracted directly from:
- summarize/CPE393_BBC_Summarization_Homework_Student.ipynb
Contains TextRank, LexRank, and BART implementations matching the course lab.
"""

from collections import Counter
import re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
from nltk.tokenize import sent_tokenize

from lexrank import STOPWORDS, LexRank


def target_sentence_count(sentences: list[str], ratio: float = 0.28) -> int:
    """
    Exact formula from Cell 7 of CPE393_BBC_Summarization_Homework_Student.ipynb.
    """
    return max(1, min(len(sentences), round(len(sentences) * ratio)))


def textrank(text: str, ratio: float = 0.28) -> list[str]:
    """
    Exact TextRank implementation following Cell 7 specifications:
    1. Sentence tokenization via sent_tokenize
    2. TfidfVectorizer with english stop_words
    3. Cosine similarity matrix with zeroed diagonal
    4. NetworkX graph from numpy array
    5. PageRank scoring
    6. Selection of top-k sentences restored to original order
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences
    
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(sentences)
    similarity = cosine_similarity(tfidf)
    np.fill_diagonal(similarity, 0)
    
    # Construct a weighted NetworkX graph from similarity
    graph = nx.from_numpy_array(similarity)
    
    # Calculate PageRank scores for all sentence indices
    scores = nx.pagerank(graph)
    
    # Choose top k indices, then put them back in original order
    k = target_sentence_count(sentences, ratio)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in ranked[:k]])
    
    return [sentences[i] for i in top_indices]


def lexrank_summary(text: str, corpus: list[list[str]] | None = None, ratio: float = 0.28, threshold: float | None = 0.25) -> list[str]:
    """
    Exact LexRank implementation following Cell 10 specifications:
    Builds LexRank instance on tokenized sentence corpus, extracts summary sentences,
    and returns them in original source order.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences
    
    k = target_sentence_count(sentences, ratio)
    if corpus is None:
        corpus = [sentences]
        
    lxr = LexRank(corpus, stopwords=STOPWORDS['en'])
    summary_sentences = lxr.get_summary(sentences, summary_size=k, threshold=threshold)
    
    # Restore original document order
    summary_set = set(summary_sentences)
    ordered_summary = [s for s in sentences if s in summary_set]
    return ordered_summary if ordered_summary else summary_sentences


def bart_summary(text: str, model=None, tokenizer=None, device=None, max_new_tokens: int = 120, min_new_tokens: int = 35) -> tuple[str, bool]:
    """
    Exact BART abstractive summarization from Cell 12:
    Uses facebook/bart-large-cnn with max 1024 token truncation check.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    MODEL_NAME = 'facebook/bart-large-cnn'
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
    elif device is None:
        device = next(model.parameters()).device

    encoded_full = tokenizer(text, truncation=False, add_special_tokens=True)
    was_truncated = len(encoded_full['input_ids']) > 1024
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4, 
            do_sample=False
        )
        
    summary_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary_text, was_truncated
