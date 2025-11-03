from inverted_index import INVERTED_INDEX
from lib.search_utils import tokenize_text
import math

def get_idf_command(term:str) -> float:

    if INVERTED_INDEX.unloaded:
        INVERTED_INDEX.load()

    num_movies = len(INVERTED_INDEX.docmap)

    term = tokenize_text(term)[0]
    index_movies = INVERTED_INDEX.index.get(term)
    
    if index_movies:
        term_movies_count = len(set(index_movies))
    else: 
        term_movies_count = 0

    return math.log((num_movies + 1) / (term_movies_count + 1))