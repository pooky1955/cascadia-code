from a2_spanbert import load_csv, load_entry
from itertools import accumulate
from nltk.tokenize import sent_tokenize
from util import load_pickle
import re


def get_drugs_sentence(annotated):
    return annotated.loc[annotated['type'].str.startswith("Drug")]


def get_advs_sentence(annotated):
    sub_df = annotated[annotated['type'].str.startswith(
        'T') & annotated['name'].str.startswith('ADE')]
    indices = [(row['value'], row['name'].split()[1:])
               for _, row in sub_df.iterrows()]
    return indices


def check_truths(advs_pos, advs_pred):
    advs_pred_set = set(advs_pred)
    all_advs = [el[0] for el in advs_pos]
    correct_advs =  [adv for adv in all_advs if adv in advs_pred_set]
    target_advs = all_advs
    return correct_advs, target_advs


def get_advs_from_pred(preds):
    return [adr for obj in preds.values() for adr in obj['ADR']]


class IndexSearcher:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(text)

    def check_aligned(self, sent, current_counter):
        for char in sent:
            if self.text[current_counter] != char:
                return False
            current_counter += 1

    def generate_mapping(self):
        current_counter = 0
        sent_mapping = dict()
        for i, sent in enumerate(self.sentences):
            while not self.check_aligned(sent, current_counter):
                assert current_counter < len(
                    self.text), "Counter has gone way over..."
                current_counter += 1
            sent_mapping[current_counter] = i
        self.sent_mapping = sent_mapping

    def get_index(self, index):
        returned_index = -1
        for key, val in self.sent_mapping.items():
            if key > index:
                return returned_index
            returned_index = val


if __name__ == "__main__":
    data = load_csv()
    groupedby = load_pickle('groupedby_n2c2.pickle')
    all_truths = []
    for i in range(len(data)):
        sample_id, raw, annotated = load_entry(data, i)
        extracted_entities = groupedby[sample_id]
        advs_pos = get_advs_sentence(annotated)
        advs_pred = get_advs_from_pred(extracted_entities)
        all_truths.append(check_truths(advs_pos, advs_pred))
