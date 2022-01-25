from functools import reduce
from typing import List, Dict
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from config import OUTPUT_DIR, MODEL_PATH, BATCH_SIZE, ADVERSE_EVENT_TAG, DRUG_EVENT_TAG
from os.path import join as pjoin
from tqdm import tqdm

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


def ner_inference(ner, text, batch_size=BATCH_SIZE):
    return [output for batch in ner(stream_raw_sentences(text), batch_size=batch_size) for output in batch]


def split_sentence(sentence):
    if len(sentence) > 512:
        return sentence.split('\n')
    else:
        return [sentence]

def stream_raw_sentences(text):
    '''Creates a generator of sentences to be fed to hf's models. along with batching + GPU , should yield significant speedup'''
    # return [''.join(sentences) for sentences in chunkify(sent_tokenize(text),SENTENCE_CHUNK_SIZE)]
    sentences = sent_tokenize(text)
    return [splitted for sentence in sentences for splitted in split_sentence(sentence)]

def word_accumulator_fn(info_dict: Dict[str, List], curr_tok):
    if curr_tok['entity'].startswith('I'):
        if len(info_dict['curr_stack']) == 0:
            return info_dict  # just don't modify it
        info_dict['curr_stack'].append(curr_tok)
    else:  # starts with B
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
        import ipdb
        ipdb.set_trace()

    if len(word_stack['curr_stack']) > 0:
        all_stacks.append(word_stack['curr_stack'])
    # little assert statement for sanity check
    clean_stacks = split_stacks(all_stacks)
    return list(filter(lambda el: el[0] != '', map(condense_stack, clean_stacks)))


def get_tokens_from_ner(raw_outputs, entity_list=['ADR', 'DRUG']):
    return {entity_type: get_tokens_from_ner_specific(raw_outputs, entity_type) for entity_type in entity_list}


def extract_info(text, ner, batch_size=BATCH_SIZE):
    '''Extracts Adverse Events and Drug names given a text (using batching and GPU)'''
    # with TimeIt("HF's NER"):
    ner_results = ner_inference(ner, text, batch_size=batch_size)
    return get_tokens_from_ner(ner_results)


def extract_entities(text, ner, batch_size=BATCH_SIZE):
    '''Extracts the entities, and return them grouped by drug and adverse events'''
    extracted_entities = extract_info(text, ner, batch_size=batch_size)
    advs = extracted_entities[ADVERSE_EVENT_TAG]
    drugs = extracted_entities[DRUG_EVENT_TAG]
    return NERResult(drugs, advs)

