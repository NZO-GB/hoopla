import json
import os
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

PUNCTUATION_TRANSTABLE = str.maketrans({key: "" for key in string.punctuation})

STEMMER = PorterStemmer()

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read()
        return set(data.splitlines())

STOPWORDS = load_stopwords()

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def str_process(text: str) -> list[str]:
    text = text.lower().translate(PUNCTUATION_TRANSTABLE).split()
    text = [STEMMER.stem(w) for w in text]
    return [w for w in text if w not in STOPWORDS]

