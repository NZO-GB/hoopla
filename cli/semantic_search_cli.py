#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model, embed_text, verify_embeddings, embed_query_text,
    SemanticSearch, chunk_text, semantic_chunk_text, ChunkedSemanticSearch
) 
from lib.search_utils import load_movies

MOVIES = load_movies()

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verify model and print max_len")

    embed_text_parser = subparsers.add_parser("embed_text", help="get embedding for text")
    embed_text_parser.add_argument("text", help="the text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verifies embeddings work")

    embedquery_parser = subparsers.add_parser("embedquery", help="generates embedding for a text")
    embedquery_parser.add_argument("query", help="the query to embed")

    semantic_search_parser = subparsers.add_parser("search", help="searches using a semantic model")
    semantic_search_parser.add_argument("query", help="the query to search")
    semantic_search_parser.add_argument("--limit", type=int, default=5, help="Max number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="splits text into chunks of n words size")
    chunk_parser.add_argument("text", help="the text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="the number of words")
    chunk_parser.add_argument("--overlap", type=int, default=40, help="the number of words to overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="splits text into chunks based on max and words size")
    semantic_chunk_parser.add_argument("text", help="the text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="the maximum size of the chunks")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="embeds chunked documents")

    args = parser.parse_args()

    match args.command:
        case "embed_chunks":
            instance = ChunkedSemanticSearch()
            embeddings = instance.load_or_create_chunk_embeddings(MOVIES)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "search":
            instance = SemanticSearch()
            instance.load_or_create_embeddings(MOVIES)
            instance.search(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()