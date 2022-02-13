from multiprocessing.sharedctypes import Value
from a2_spanbert import load_csv, load_entry
from tqdm import tqdm
import gc
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer

def get_advs_sentence(annotated):
    sub_df = annotated[annotated['type'].str.startswith(
        'T') & annotated['name'].str.startswith('ADE')]
    indices = [row['name'].split()[1:]
               for _, row in sub_df.iterrows()]
    inted_indices = [(int(el[0]),int(el[-1])) for el in indices]
    return inted_indices

def resolve_advs(advs_spans,sent_spans):
    spans_ans = []
    for start_adv, end_adv in advs_spans:
        for sent_id,(start_sent, end_sent) in enumerate(sent_spans):
            if start_sent <= start_adv <= end_sent and start_sent <= end_adv <= end_sent:
                spans_ans.append(sent_id)
                break
    return spans_ans

def find_span(raw,sent_spans,ind):
    start,end = sent_spans[ind]
    return raw[start:end]

def extract_spans(raw,sent_spans,inds):
    return [find_span(raw,sent_spans,ind) for ind in inds]


if __name__ == "__main__":
    pkt = PunktSentenceTokenizer()
    # data = load_csv()
    data = load_csv('n2c2_test.csv')
    all_related_sents = []
    all_unrelated_sents = []
    for i in tqdm(range(len(data))):
        sample_id, raw, annotated = load_entry(data,i)
        advs_spans = get_advs_sentence(annotated)
        sent_spans = list(pkt.span_tokenize(raw))
        related_sents_inds = resolve_advs(advs_spans,sent_spans)
        related_sents_inds_set = set(related_sents_inds)
        unrelated_sents_inds = [i for i in range(len(sent_spans)) if i not in related_sents_inds_set]

        related_sents = extract_spans(raw,sent_spans,related_sents_inds)
        unrelated_sents = extract_spans(raw,sent_spans,unrelated_sents_inds)

        all_related_sents.extend(related_sents)
        all_unrelated_sents.extend(unrelated_sents)

        gc.collect()
        

    labels = [1 for _ in range(len(all_related_sents))] + [0 for _ in range(len(all_unrelated_sents))]
    df = pd.DataFrame()
    df['label'] = labels
    df['text'] = all_related_sents + all_unrelated_sents
    # df.to_csv("./data/intermediate/ade_related_unrelated.csv")
    df.to_csv("./data/intermediate/ade_related_unrelated_test.csv")
