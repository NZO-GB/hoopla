import numpy as np
import os
from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    load_movies, cosine_similarity, EMBEDDINGS_PATH, CHUNK_EMBEDDINGS_PATH, JSON_METADATA_PATH
) 
import re
import json

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):

        if text.strip() == "":
            raise ValueError("text is empty")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def _populate_docs_and_docmap(self, documents):
        self.documents = documents
        movie_list = []
        for doc in documents:
            key = doc["id"]
            self.document_map[key] = doc
            title_description = f"{doc['title']}: {doc['description']}"
            movie_list.append(title_description)
        return movie_list
    
    def _encode_embeddings(self, movie_list):
        self.embeddings = self.model.encode(movie_list, show_progress_bar = True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        
    
    def build_embeddings(self, documents):
        movie_list = self._populate_docs_and_docmap(documents)
        self._encode_embeddings(movie_list)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        movie_list = self._populate_docs_and_docmap(documents)
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        self._encode_embeddings(movie_list)
        return self.embeddings
    
    def search(self, query, limit):
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
        else:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embedding = self.generate_embedding(query)

        values_list = []
        for id, doc in self.document_map.items():
            m_embedding = self.embeddings[id-1]
            score = cosine_similarity(m_embedding, q_embedding)
            title, description = doc["title"], doc["description"][:100] + "..."
            values_list.append({
                "score": score, "title": title, "description": description}
                )
        sorted_list = sorted(values_list, key=lambda x: x["score"], reverse=True)
        for i in range(0, limit):
            print(f"{i+1}. {sorted_list[i]["title"]} (score: {sorted_list[i]["score"]})\n{sorted_list[i]["description"]}")

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> list[list[int]]:
        self._populate_docs_and_docmap(documents)
        chunks: list[str] = []
        metadata: list[dict] = []
        for document in documents:
            if len(document["description"]) == 0:
                continue
            chunked_doc = semantic_chunk_text(document["description"], 4, 1, False)
            len_chunked_doc = len(chunked_doc)
            for i, chunk in enumerate(chunked_doc):
                chunks.append(chunk)
                chunk_metadata = {
                    "movie_idx": document["id"],
                    "chunk_idx": i,
                    "total_chunks": len_chunked_doc
                }
                metadata.append(chunk_metadata)
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(JSON_METADATA_PATH, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._populate_docs_and_docmap(documents)
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(JSON_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(JSON_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

def verify_embeddings():
    instance = SemanticSearch()
    documents = load_movies()
    instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {instance.embeddings.shape[0]} vectors in {instance.embeddings.shape[1]} dimensions")
    print(instance.embeddings)

def embed_text(text):
    Semantic_instance = SemanticSearch()
    embedding = Semantic_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    instance = SemanticSearch()
    print(f"Model loaded: {instance.model}")
    print(f"Max sequence length: {instance.model.max_seq_length}")

def embed_query_text(query, debug = True):
    Semantic_instance = SemanticSearch()
    embedding = Semantic_instance.generate_embedding(query)
    if debug:
        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}")

def chunk(text, max, overlap, debug = True):
    chunks = []
    start = 0
    end = max
    while start + overlap < len(text):
        raw_append = (text[start:end]) 
        chunks.append(raw_append)
        start += max-overlap
        end += max-overlap
    if debug:
        for i, chunk in enumerate(chunks, 1):
            chunk_text = ' '.join(chunk)
            print(f"{i}. {chunk_text}")
    return chunks

def chunk_text(text, n, overlap, debug = True):
    characters = len(text)
    if debug:
        print(f"Chunking {characters} characters")
    text = text.split()
    chunk(text, n, overlap, debug)
    
def semantic_chunk_text(text, max, overlap, debug = True):
    characters = len(text)
    text = re.split(r"(?<=[.!?])\s+", text)
    print(f"Semantically chunking {characters} characters")
    return chunk(text, max, overlap, debug)

