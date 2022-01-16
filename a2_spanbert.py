from functools import reduce
from select import select
from typing import List, Dict
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
from config import OUTPUT_DIR, MODEL_PATH, BATCH_SIZE, ADVERSE_EVENT_TAG, DRUG_EVENT_TAG
from os.path import join as pjoin
from io import StringIO
from util import TimeIt, chunkify

# transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import namedtuple

MimicNote = namedtuple("MimicNote", "sample_id raw_text annotated_tsv")
NERResult = namedtuple("NERResult", "drugs advs")

# importing the models


def load_ner(use_gpu=True, model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    if use_gpu:
        return pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    else:
        return pipeline("ner", model=model, tokenizer=tokenizer)


def load_csv(path="n2c2.csv", directory=OUTPUT_DIR):
    '''Loads the n2c2 csv. Columns look like |sample_id|raw|annotated_tsv|'''
    csv_file = pjoin(directory, path)
    df = pd.read_csv(csv_file)
    return df


def ner_inference(ner, text, batch_size=BATCH_SIZE):
    return [output for batch in ner(stream_raw_sentences(text), batch_size=batch_size) for output in batch ]


def load_entry(df, index):
    '''Returns a Named Tuple containing the raw_text, annotated_tsv, and sample_id'''
    row = df.iloc[index]
    sample_id = row['sample_id']
    raw_text = row['raw']
    annotated_tsv = pd.read_csv(StringIO(
        row['annotated']), sep='\t', header=None, names=['type', 'name', 'value'])
    return MimicNote(sample_id, raw_text, annotated_tsv)


def stream_raw_sentences(text):
    '''Creates a generator of sentences to be fed to hf's models. along with batching + GPU , should yield significant speedup'''
    # return [''.join(sentences) for sentences in chunkify(sent_tokenize(text),SENTENCE_CHUNK_SIZE)]
    return sent_tokenize(text)


def verify_sub_words_are_consecutive(stack):
    indices = [el['index'] for el in stack]
    try:
        assert reduce(lambda holds_true_prev, curr_pair: holds_true_prev and (
            curr_pair[0] + 1 == curr_pair[1]), zip(indices, indices[1:]), True), "Not consecutive!"
    except AssertionError:
        import ipdb; ipdb.set_trace()


def word_accumulator_fn(info_dict: Dict[str, List], curr_tok):
    if curr_tok['entity'].startswith('I'):
        if len(info_dict['curr_stack']) == 0:
            return info_dict  # just don't modify it
        info_dict['curr_stack'].append(curr_tok)
    else:  # starts with B
        assert curr_tok['entity'].startswith(
            'B'), "THERE IS SOME SUSSY STUFF GOING ON WITH THE TAGGER. OUTPUT OTHER THAN I- OR B-???"
        info_dict['word_list'].append(info_dict['curr_stack'])
        info_dict['curr_stack'] = [curr_tok]
    return info_dict


def condense_stack(stack):
    condensed_word = ''.join([part['word'][2:] if part['word'].startswith(
        "##") else ' ' + part['word'] for part in stack]).lstrip()
    scores = np.array([part['score'] for part in stack])
    return condensed_word, scores


def split_one_stack(stack):
    '''Processes only 1 stack and returns a list of untangled stacks (1 stack = 1 entity)'''
    # split section of indices into partitions
    current_ind = -1000
    partitions = []
    current_partition = []
    for el in stack:
        if el['index'] != current_ind + 1:
            # time for new partition
            if len(current_partition) != 0:
                partitions.append(current_partition)
            current_partition = [el]
        else:
            current_partition.append(el)
        current_ind = el['index']
    
    if len(current_partition) != 0:
        partitions.append(current_partition)
    
    return partitions



def split_stacks(stacks):
    '''Processes the stacks to return the correct entity stacks. (Untangles multiple entities in 1 stack to ensure each stack = 1 entity)'''
    # sometimes there will be multiple "entities" grouped in 1 stack but with indices that are not contiguous
    return [stack_splitted for stack in stacks for stack_splitted in split_one_stack(stack)]


def get_tokens_from_ner_specific(raw_outputs, entity_type):
    '''gets proper tokens from nlp pipeline (merges subwords as well)'''
    # output is a list of {entity : str, score : float, index : int, word : str, start : int, end : int}
    # goal: merge consecutive indices into one word
    outputs = filter(lambda output: output['entity'].endswith(
        entity_type), raw_outputs)
    try:
        word_stack = reduce(word_accumulator_fn, outputs,
                            dict(word_list=[], curr_stack=[]))
        all_stacks = word_stack['word_list']
    except Exception as e:
        print("exception:", str(e))
        import ipdb; ipdb.set_trace()

    if len(word_stack['curr_stack']) > 0:
        all_stacks.append(word_stack['curr_stack'])
    # little assert statement for sanity check
    clean_stacks = split_stacks(all_stacks)
    [verify_sub_words_are_consecutive(stack) for stack in clean_stacks]
    return list(filter(lambda el: el[0] != '', map(condense_stack, clean_stacks)))


def get_tokens_from_ner(raw_outputs, entity_list=['ADR', 'DRUG']):
    return {entity_type: get_tokens_from_ner_specific(raw_outputs, entity_type) for entity_type in entity_list}


def extract_info(text, ner, batch_size=BATCH_SIZE):
    '''Extracts Adverse Events and Drug names given a text (using batching and GPU)'''
    with TimeIt("HF's NER"):
        ner_results = ner_inference(ner, text, batch_size=batch_size)
    return get_tokens_from_ner(ner_results)



def extract_entities(text, ner, batch_size=BATCH_SIZE):
    '''Extracts the entities, and return them grouped by drug and adverse events'''
    extracted_entities = extract_info(text, ner, batch_size=batch_size)
    advs = extracted_entities[ADVERSE_EVENT_TAG]
    drugs = extracted_entities[DRUG_EVENT_TAG]
    return NERResult(drugs, advs)


'''
-----------------------------
Code for processing the ground truth (annotated tsv)
-----------------------------
'''


def is_drug(row):
    return row['name'].startswith("Drug") and row['type'].startswith("T")


def is_adv(row):
    return row['name'].startswith("ADE") and row['type'].startswith("T")


def extract_drug_names_truth(annotated_tsv):
    return [row['value'].lower() for _, row in annotated_tsv.iterrows() if is_drug(row)]


def extract_adv_names_truth(annotated_tsv):
    return [row['value'].lower() for _, row in annotated_tsv.iterrows() if is_adv(row)]

def select_words_by_thresh(extracted_entities,filter_fn):
    return [term for term , scores in extracted_entities if filter_fn(scores)] 

def observe_differences(pred_set,true_set):
    print(f"Intersection:")
    print(pred_set.intersection(true_set))
    print(f"Extra in Preds:")
    print(pred_set.difference(true_set))
    print(f"Missed in Preds:")
    print(true_set.difference(pred_set))

if __name__ == "__main__":
    with TimeIt("Loading CSV dataset"): dataset = load_csv()
    with TimeIt("Loading NER model"): ner = load_ner(use_gpu=True)
    sample_index = 2
    sample_entry = load_entry(dataset,sample_index)
    raw_text = sample_entry.raw_text
    annotated_tsv = sample_entry.annotated_tsv

    with TimeIt("performing NER"): r_pred_drugs, r_pred_advs = extract_entities(raw_text,ner)
    score_fn = lambda scores : scores.mean() > 0.7
    pred_drugs = select_words_by_thresh(r_pred_drugs,score_fn)
    pred_advs = select_words_by_thresh(r_pred_advs,score_fn)
    
    true_drugs = extract_drug_names_truth(annotated_tsv)
    true_advs = extract_adv_names_truth(annotated_tsv)

    pred_drugs_set, pred_advs_set, true_drugs_set, true_advs_set = map(set,[pred_drugs,pred_advs,true_drugs,true_advs])
    observe_differences(pred_drugs_set,true_drugs_set)
    observe_differences(pred_advs_set,true_advs_set)

    

