from inverted_index import INVERTED_INDEX
from lib.get_tf import get_tf_command
from lib.search_utils import tokenize_text, BM25_K1, BM25_B
import math


def get_bm25_idf_command(term:str) -> float:

    if INVERTED_INDEX.unloaded:
        INVERTED_INDEX.load()

    num_movies = len(INVERTED_INDEX.docmap)
    term = tokenize_text(term)
    if len(term) > 1:
        raise Exception("Invalid token: more than one word")
    term = term[0]

    df = len(INVERTED_INDEX.get_documents(term))

    result = math.log((num_movies - df + 0.5) / (df + 0.5) + 1)
    return result

def get_bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:

    if INVERTED_INDEX.unloaded:
        INVERTED_INDEX.load()

    avg_len = INVERTED_INDEX.avg_doc_length
    doc_len = INVERTED_INDEX.doc_lengths.get(doc_id)

    len_norm = 1 - b + b * (doc_len / avg_len)

    tf = get_tf_command(doc_id, term)
    saturation = (tf * (k1 + 1)) / (tf + k1 * len_norm)

    return saturation

