import numpy as np
import os
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies, cosine_similarity, EMBEDDINGS_PATH

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
    
    def __populate_docs_and_docmap(self, documents):
        self.documents = documents
        movie_list = []
        for doc in documents:
            key = doc["id"]
            self.document_map[key] = doc
            title_description = f"{doc['title']}: {doc['description']}"
            movie_list.append(title_description)
        return movie_list
    
    def __encode_embeddings(self, movie_list):
        self.embeddings = self.model.encode(movie_list, show_progress_bar = True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        
    
    def build_embeddings(self, documents):
        movie_list = self.__populate_docs_and_docmap(documents)
        self.__encode_embeddings(movie_list)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        movie_list = self.__populate_docs_and_docmap(documents)
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        self.__encode_embeddings(movie_list)
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

def embed_query_text(query):
    Semantic_instance = SemanticSearch()
    embedding = Semantic_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def chunk_text(text, n, overlap):
    characters = len(text)
    text = text.split()
    chunks = []
    start = 0
    end = n
    while start + overlap < len(text):
        raw_append = (text[start:end]) 
        chunks.append(raw_append)
        start += n-overlap
        end += n-overlap
    print(f"Chunking {characters} characters")
    for i, chunk in enumerate(chunks, 1):
        chunk_text = ' '.join(chunk)
        print(f"{i}. {chunk_text}")
