from util import load_pickle
from config import MIMIC_NOTES_PICKLE_SHUFFLED, RXNAV_PATH, MEDDRA_CSV_PATH
import pandas as pd
from util import load_pickle
from a4_rxnorm import RxNav
from collections import Counter
from itertools import groupby
from Levenshtein import ratio
import re
from collections import Counter
from string import punctuation

groupedby = load_pickle("groupedby.pickle")

punc_patt = re.compile(rf'[{punctuation}]')
anti_word_patt = re.compile(rf'\W')

all_drugs = {drug for obj in groupedby.values() for pred in obj.values()
             for drug in pred['DRUG']}
all_drugs_len = [len(drug.split()) for drug in all_drugs]

sorted_drugs = list(sorted(all_drugs, key=lambda drug: len(drug.split())))
bylens_group = {length: list(els) for length, els in groupby(
    sorted_drugs, key=lambda drug: len(drug.split()))}

rxnav: RxNav = load_pickle(RXNAV_PATH)
rx_df, drug_targets, rx_str_id, rx_id_str, rx_resolve = rxnav

meddra_csv = pd.read_csv(MEDDRA_CSV_PATH)

all_meddra_drugs = set(meddra_csv['drug_name'].unique())



def find_with_exceptions(drug_name):
    if drug_name.startswith("ancef"):
        return (1, "cefazolin")
    return None


def find_closest_drug(drugname):
    first_word = drugname.split()[0]
    scores = ((ratio(first_word.lower(), target_drug.lower()), target_drug)
              for target_drug in all_meddra_drugs)
    return max(scores, key=lambda el: el[0])


def find_closest_drug_rx(drugname):
    first_word = drugname.split()[0]
    scores = ((ratio(first_word.lower().split()[0], target_drug.lower().split()[
              0]), target_drug) for target_drug in drug_targets)
    # maximize ratio, then minimize length
    return max(scores, key=lambda el: (el[0], -len(el[1])))


def extract_drug_sider(drug_name):
    candidates = [find_closest_drug(el) for el in anti_word_patt.sub(
        ' ', drug_name).split() if len(el) > 3]
    return [term for score, term in candidates if score > 0.8]


def extract_drug_rx(drug_name):
    all_drugs = punc_patt.split(drug_name)
    candidates = [find_rxnav(el) for drug_name in all_drugs for el in anti_word_patt.sub(
        ' ', drug_name).split() if len(el) > 3]
    return candidates


def find_rxnav(term):
    likely_can = [(target, rx_str_id[target])
                  for target in drug_targets if target.lower().startswith(term.lower())]
    return min(likely_can, key=lambda target: len(target[0]), default=None)


def find_rxnav_all(term):
    likely_can = [(target, rx_str_id[target])
                  for target in drug_targets if target.lower().startswith(term.lower())]
    return sorted(likely_can, key=lambda target: len(target[0]))


def find_sider_from_rxnav_with_id(rx_id):
    if rx_id not in rx_resolve:
        return None
    sider_possible_set = {term.split()[0].lower()
                          for _, term in rx_resolve[rx_id]}
    scores = [find_closest_drug(sider_possible)
              for sider_possible in sider_possible_set]
    return max(scores, key=lambda el: el[0], default=None)


def find_sider_from_rxnav(term):
    rx_id_result = find_rxnav(term)
    if rx_id_result is None:
        return None
    _, rx_id = rx_id_result
    return find_sider_from_rxnav_with_id(rx_id)


def find_closest_sider_from_rxnav(term):
    score, term = find_closest_drug_rx(term)
    if score < 0.8:
        return None
    rx_id = rx_str_id[term]
    return find_sider_from_rxnav_with_id(rx_id)


def match_drug(drug_name):
    '''This is the function to use when matching drugs'''
    # first check if it's exception
    fns = [find_with_exceptions, find_closest_drug,
           find_sider_from_rxnav, find_closest_sider_from_rxnav]
    for fn in fns:
        result = fn(drug_name)
        if result is not None:
            score, _ = result
            if score > 0.8:
                return result