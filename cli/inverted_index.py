from lib.search_utils import tokenize_text, load_movies
from collections import Counter
import pickle
import os

MOVIES = load_movies()

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[str, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length = 0
        self.unloaded = True

    def __add_document(self, doc_id: int, text: str):
        text = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(text)
        self.doc_lengths[doc_id] = len(text)
        for word in text:
            if word not in self.index:
                self.index[word] = []
            if doc_id not in self.index[word]:
                self.index[word].append(doc_id)

    def __get_avg_doc_length(self) -> float:

        if len(self.doc_lengths) == 0:
            return 0.0
        total = 0
        for length in self.doc_lengths.values():
            total += length

        return total / len(self.doc_lengths)
    
    def get_documents(self, term):
        try:
            result = sorted(self.index.get(term.lower()))
        except TypeError:
            return 
        return result
        
    def build(self):
        for m in MOVIES:
            title_description = f"{m["title"]} {m["description"]}"
            self.__add_document(m["id"], title_description)
            self.docmap[m["id"]] = m

        self.save()

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):

        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
        except:
            print("Unable to find index file")
        try:
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
        except:
            print("Unable to find docmap file")
        
        try: 
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except:
            print("Unable to find term frequencies")

        try:
            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
        except:
            print("Unable to find doc lengths")

        if self.unloaded:
            self.avg_doc_length = self.__get_avg_doc_length()

        self.unloaded = False

INVERTED_INDEX = InvertedIndex()