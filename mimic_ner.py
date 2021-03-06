# -*- coding: utf-8 -*-
"""MIMIC_NER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vSCfx6dHw9wtO6qQTw7icJsYSYbYYuBL
"""
from torch.utils.data import Dataset
from string import punctuation
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import sent_tokenize
from config import MIMIC_NOTES_PICKLE_SHUFFLED, MODEL_PATH
from os.path import join as pjoin
from tqdm import tqdm
import pandas as pd
# transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import namedtuple
import nltk
import re
# bam
from tqdm import tqdm
import gc

import os
from util import save_pickle, load_pickle

# for NLP treatment
eng_stopwords = set(stopwords.words('english'))
unused_patt = re.compile(rf"\[\*\*[^\*]+\*\*]")
# [** hello **]
mimic_notes = pd.read_pickle(MIMIC_NOTES_PICKLE_SHUFFLED)
SIZE = 10000
# importing the models


def clean_sent(text):
    no_patt_text = unused_patt.sub(" ", text.lower())
    clean_toks = [tok for tok in no_patt_text.split()
                  if tok not in eng_stopwords]
    return ' '.join(clean_toks)


def load_ner(use_gpu=True, model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    if use_gpu:
        return pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    else:
        return pipeline("ner", model=model, tokenizer=tokenizer)


def stream_raw_sentences(text):
    '''Creates a generator of sentences to be fed to hf's models. along with batching + GPU , should yield significant speedup'''
    return [sentence[:512] for sentence in sent_tokenize(clean_sent(text))]


class TextDataset(Dataset):
    def __init__(self, start, end):
        global mimic_notes
        df = mimic_notes[start:end].reset_index()
        self.df = df
        self.texts = self.df['TEXT']
        self.ids = self.df['ROW_ID']

    def ready(self):
        self.all_sentences = [(i, stream_raw_sentences(text))
                              for i, text in enumerate(self.texts)]
        self.all_sentences_flat = [
            (i, sentence, j) for i, sentences in self.all_sentences for j, sentence in enumerate(sentences)]
        self.all_sentences_flat.sort(key=lambda x: len(x[1]))
        self.indices_mapping = [(el[0], el[2])
                                for el in self.all_sentences_flat]
        self.row_id_mappings = [(self.ids[i], j)
                                for i, j in self.indices_mapping]
        self.inputs = [el[1] for el in self.all_sentences_flat]

    def __getitem__(self, index):
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)


def generate_datasets(size):
    global mimic_notes
    return (TextDataset(i, i+size) for i in range(0, len(mimic_notes), size))


def fit(ner):
    for i, dataset in enumerate(generate_datasets(SIZE)):
        if i < 69:
            continue
        dataset.ready()
        dirname = f"data/output/data-{i}-{SIZE}"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        save_pickle(dataset.row_id_mappings, "id_mappings.pickle", dir=dirname)
        preds = [batch for batch in tqdm(ner(
            dataset, num_workers=4, batch_size=64), desc=f"Predicting for EPOCH {i}", total=len(dataset))]
        save_pickle(preds, "preds.pickle", dir=dirname)
        print(f"Finished chunk {i}")
        gc.collect()
        # break


if __name__ == "__main__":
    ner = load_ner(use_gpu=True)
    fit(ner)
