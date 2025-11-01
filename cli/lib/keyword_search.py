from lib.search_utils import(DEFAULT_SEARCH_LIMIT, tokenize_text)
from inverted_index import INVERTED_INDEX

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    INVERTED_INDEX.load()
    results = set()
    query = tokenize_text(query)
    
    for token in query:
        result = INVERTED_INDEX.get_documents(token)
        if result:
            for index in result:
                if len(results) >= limit:
                    return index_to_movie(results)
                results.add(index)
        
    return index_to_movie(results)

def index_to_movie(results: list[int]) -> list[dict]:
    movies: list[dict] = []
    for index in results:
        movies.append(INVERTED_INDEX.docmap[index])
    return movies