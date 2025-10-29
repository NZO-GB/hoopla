from inverted_index import INVERTED_INDEX
from lib.search_utils import str_process
import math


def get_bm25_idf(term:str) -> float:

    INVERTED_INDEX.load()

    num_movies = len(INVERTED_INDEX.docmap)
    term = str_process(term)
    if len(term) > 1:
        raise Exception("Invalid token: more than one word")
    term = term[0]

    df = len(INVERTED_INDEX.get_documents(term))

    result = math.log((num_movies - df + 0.5) / (df + 0.5) + 1)
    return result