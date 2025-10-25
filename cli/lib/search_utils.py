import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

PUNCTUATION_TRANSTABLE = str.maketrans({key: "" for key in string.punctuation})

STOPWORDS = 

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def str_process(text: str) -> list[str]:
    return text.lower().translate(PUNCTUATION_TRANSTABLE).split()