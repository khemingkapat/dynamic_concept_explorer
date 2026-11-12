"""
Baseline topic modeling module extracted directly from:
- topic_modelling/bbc_topic_modeling.ipynb
Contains preprocessing, Gensim BoW dictionary/corpus construction, and LDA training.
"""

import nltk
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models

# Download WordNet data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> list[str]:
    """
    Exact text preprocessing function from Cell 4 of bbc_topic_modeling.ipynb:
    1. Tokenize and normalize (min_len=3, deacc=True)
    2. Stopword removal and lemmatization (verb + noun)
    """
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    cleaned = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos="v"), pos="n")
        for word in tokens
        if word not in STOPWORDS
    ]
    return cleaned


def build_corpus_and_dictionary(tokenized_docs: list[list[str]], no_below: int = 5, no_above: float = 0.5):
    """
    Exact dictionary filtering and BoW corpus generation from Cell 6 of bbc_topic_modeling.ipynb.
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
    return dictionary, corpus


def train_lda(corpus, dictionary, num_topics: int, passes: int = 10, random_state: int = 42):
    """
    Exact LDA training function from Cell 8 of bbc_topic_modeling.ipynb.
    """
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )


def print_model_topics(model, num_words: int = 8) -> list[tuple[int, str]]:
    """
    Exact topic printing function from Cell 8 of bbc_topic_modeling.ipynb.
    """
    topics = []
    for topic_id, topic_str in model.print_topics(num_topics=model.num_topics, num_words=num_words):
        topics.append((topic_id, topic_str))
    return topics
