import spacy
import scispacy
from scispacy.linking import EntityLinker
from config import N2C_DIR
from os.path import join as pjoin
from util import read_file, save_pickle

pipeline = spacy.load("en_core_sci_sm")
pipeline.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
text = read_file("108809",dir=N2C_DIR,ext="txt")
result = pipeline(text)
save_pickle(result,"result.pickle")

