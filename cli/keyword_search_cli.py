#!/usr/bin/env python3
from inverted_index import INVERTED_INDEX
import argparse

from lib.search_utils import BM25_K1, BM25_B
from lib.get_idf import get_idf_command
from lib.get_tf import get_tf_command
from lib.get_tfidf import get_tfidf_command
from lib.get_bm25 import get_bm25_idf_command, get_bm25_tf_command
from lib.keyword_search import(
    search_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="returns frequency of term in movie given by doc_id")
    tf_parser.add_argument("doc_id", type=str, help="doc_id of the movie to search")
    tf_parser.add_argument("term", type=str, help="the therm to get the frequency of")

    idf_parser = subparsers.add_parser(
        "idf", help="returns the inversed document frequency of a term")
    idf_parser.add_argument("term", type=str, help="the term to get the frequency of")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="returns the tf-idf of a movie given by doc_id")
    tfidf_parser.add_argument("doc_id", type=str, help="doc_id of the movie to search")
    tfidf_parser.add_argument("term", type=str, help="the therm to get the tf-idf of")

    bm25_idf_parser = subparsers.add_parser(
      'bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term")
    
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    args = parser.parse_args()
    
    match args.command:
        case "bm25tf":
            bm25tf = get_bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}': {bm25tf:.2f}")
        case "bm25idf":
            bm25idf = get_bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "tfidf":
            tfidf = get_tfidf_command(int(args.doc_id), args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case "idf":
            idf = get_idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
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
