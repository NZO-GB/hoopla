#!/usr/bin/env python3
from inverted_index import INVERTED_INDEX
import argparse

from lib.keyword_search import(
    search_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="build the inverted index")

    search_parser = subparsers.add_parser("save", help="save the inverted index to file")

    args = parser.parse_args()
    
    match args.command:
        case "build":
            INVERTED_INDEX.build()
        case "save":
            INVERTED_INDEX.save()
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f" {i}. {res["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
