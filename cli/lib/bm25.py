from inverted_index import INVERTED_INDEX
from lib.search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text
from lib.get_bm25 import get_bm25_tf_command, get_bm25_idf_command

def bm25(doc_id: int, term: str) -> float:
    tf = get_bm25_tf_command(doc_id, term)
    idf = get_bm25_idf_command(term)
    return tf * idf

def bm25_search(query: str, limit = DEFAULT_SEARCH_LIMIT) -> list:

    if INVERTED_INDEX.unloaded:
        INVERTED_INDEX.load()

    query = tokenize_text(query)
    scores = {}
    for doc_id in INVERTED_INDEX.docmap:
        total = 0
        for token in query:
            total += bm25(doc_id, token)
        scores[doc_id] = total
    
    sorted_dict = sorted(scores.items(), key=lambda x: x[1], reverse = True)
    
    result = []
    for i in range(limit):
        doc = sorted_dict[i]
        doc_id, score = doc[0], doc[1]
        doc_name = INVERTED_INDEX.docmap[doc_id]["title"]
        result.append((doc_id, doc_name, score))
    
    return result
        