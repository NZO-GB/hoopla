from inverted_index import INVERTED_INDEX
from lib.get_tf import get_tf_command
from lib.get_idf import get_idf_command

def get_tfidf_command(doc_id:int, term:str) -> float:

    INVERTED_INDEX.load()

    tf = get_tf_command(doc_id, term)
    idf = get_idf_command(term)

    tfidf = tf * idf

    return tfidf