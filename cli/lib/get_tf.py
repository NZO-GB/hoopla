from inverted_index import INVERTED_INDEX
from lib.search_utils import str_process

def get_tf_command(doc_id: str, term: str) -> int:
    INVERTED_INDEX.load()
    term = str_process(term)
    if len(term) > 1:
        raise Exception ("Can't get frequency of more than 1 term")
    term = term[0]
    return INVERTED_INDEX.term_frequencies[doc_id][term]