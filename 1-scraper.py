'''
Script 1 
---------
Reads N2C2 Directory (expected to be in data/n2c_training)
Saves a pickle object of lists of named tuples, (acts as dataset)
Also saves a complete pandas dataframe for loading afterwards
'''

#-------- Python Packages -----#
import os
from os.path import join as pjoin
from tqdm import tqdm
import pickle
import pandas as pd
from collections import namedtuple

#------- My own custom packages -----#
from config import N2C_DIR, OUTPUT_DIR
from util import save_pickle, read_file

DataSample = namedtuple('DataSample', 'sample_id raw annotated')



def get_ids(dirname=N2C_DIR):
    return (filename.split(".")[0] for filename in os.listdir(dirname) if filename.endswith(".txt"))


def read_annotated_dir(dirname=N2C_DIR):
    for n2c_id in tqdm(get_ids(dirname), desc="Processing IDs"):
        annotated = read_file(n2c_id, ext="ann", dir=dirname)
        raw = read_file(n2c_id, ext="txt", dir=dirname)
        yield DataSample(sample_id=n2c_id, raw=raw, annotated=annotated)


all_dirs = list(read_annotated_dir())
save_pickle(all_dirs, "data.pickle")
pd.DataFrame(all_dirs).to_csv(pjoin(OUTPUT_DIR, "n2c2.csv"))
print("saved :)")
