# standard packages
import pickle
from os.path import join as pjoin

# own packages
from config import OUTPUT_DIR


def save_pickle(obj, filename, dir=OUTPUT_DIR):
    with open(pjoin(dir, filename), "wb") as f:
        
        pickle.dump(obj, f)


def load_pickle(filename, dir=OUTPUT_DIR):
    with open(pjoin(dir, filename), "rb") as f:
        return pickle.load(f)

def read_file(filename, dir=None, ext=None):
    filename = f"{filename}.{ext}" if ext is not None else filename
    path = filename if dir is None else pjoin(dir, filename)
    with open(path, "r") as f:
        return f.read()

