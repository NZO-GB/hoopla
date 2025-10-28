#!/usr/bin/env python3
from inverted_index import INVERTED_INDEX
import argparse

from lib.get_tf import get_tf_command
from lib.keyword_search import(
    search_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="build the inverted index")

    search_parser = subparsers.add_parser("tf", help="returns frequency of term in doc given by doc_id")
    search_parser.add_argument("doc_id", type=str, help="doc_id of the movie to search")
    search_parser.add_argument("term", type=str, help="the therm to get the frequency of")



    args = parser.parse_args()
    
    match args.command:
        case "build":
            INVERTED_INDEX.build()
        case "tf":
            print(f"Getting the frequency in doc '{args.doc_id}' of term '{args.term}'")
            result = get_tf_command(int(args.doc_id), args.term)
            print(result)
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f" {i}. {res["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
