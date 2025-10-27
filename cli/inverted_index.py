from lib.search_utils import str_process, load_movies
from collections import Counter
import pickle
import os

MOVIES = load_movies()

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: Counter = Counter()

    def __add_document(self, doc_id, text):
        text = str_process(text)
        for word in text:
            if word not in self.index:
                self.index[word] = []
            self.index[word].append(doc_id)
    
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
        

INVERTED_INDEX = InvertedIndex()