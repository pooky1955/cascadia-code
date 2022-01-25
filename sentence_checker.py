import pandas as pd
from tqdm import tqdm
from config import MIMIC_NOTES_PICKLE_SHUFFLED
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from util import save_pickle
import re
eng_stopwords = set(stopwords.words('english'))
unused_patt = re.compile(rf"\[\*\*[^\*]+\*\*]")

# importing the models

def clean_sent(text):
    no_patt_text = unused_patt.sub(" ", text.lower())
    clean_toks = [tok for tok in no_patt_text.split()
                  if tok not in eng_stopwords]
    return ' '.join(clean_toks)

def check_over_sentences(text):
    '''Creates a generator of sentences to be fed to hf's models. along with batching + GPU , should yield significant speedup'''
    return [(j,sentence) for j,sentence in enumerate(sent_tokenize(clean_sent(text))) if len(sentence) > 512]

if __name__ == "__main__":
    df = pd.read_pickle(MIMIC_NOTES_PICKLE_SHUFFLED)
    row_ids = df['ROW_ID'].values
    all_sents = [check_over_sentences(text) for text in tqdm(df['TEXT'].values)]
    nonzeros = [(i,row_ids[i],j,sent) for i, (j,sent) in enumerate(all_sents) if len(sent[1]) != 0]
    save_pickle(nonzeros,"non_zeros_sents.pickle")