from inverted_index import INVERTED_INDEX
from search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text
from get_bm25 import get_bm25_tf_command, get_bm25_idf_command

def bm25(doc_id: int, term: str) -> float:
    tf = get_bm25_tf_command(doc_id, term)
    idf = get_bm25_idf_command(doc_id, term)
    return tf * idf

def bm25_search(query: str, limit = DEFAULT_SEARCH_LIMIT) -> list:
    query = tokenize_text(query)
    scores = {}
    for doc_id in INVERTED_INDEX.docmap.keys():
        total = 0
        for token in query:
            total += bm25(doc_id, token)
        scores.update(doc_id, total)
    
    
        