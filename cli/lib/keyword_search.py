from .search_utils import(DEFAULT_SEARCH_LIMIT, load_movies, str_process)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query = str_process(query)
    for movie in movies:
        title = str_process(movie["title"])
        if match(query, title):
            results.append(movie)
        if len(results) >= limit:
            break
    return results

def match(query: list[str], title: list[str]) -> bool:
    for word_q in query:
        for word_t in title:
            if word_q in word_t:
                return True