'''Finds all entity in all texts'''
import pandas as pd
import spacy
import scispacy
from scispacy.linking import EntityLinker

CHUNK_SIZE = 1000
df = pd.read_pickle("notes.pickle")
pipeline = spacy.load("en_core_sci_sm")
pipeline.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

def save_pickle(obj, name):
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(obj, f)


def chunka_chunka(values_list, size):
    for i in range(0,len(values_list),size):
        yield values_list[i:i+size]


def do_it_chunky(chunk, i):
    objs = [pipeline(text) for text in chunk]
    save_pickle(objs, f"data/chunks-{CHUNK_SIZE}/chunk_{i}")


# go over each text, and perform entity recognition
for i, text in enumerate(chunka_chunka(df["TEXT"].values, CHUNK_SIZE)):
    do_it_chunky(text, i)
    import ipdb; ipdb.set_trace()
    break
