# standard packages
import pickle
from os.path import join as pjoin
from time import time

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


def chunkify(values_list, size):
    for i in range(0, len(values_list), size):
        yield values_list[i:i+size]


class TimeIt:
    def __init__(self, prompt):
        self.prompt = prompt
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, *_):
        self.end_time = time()
        time_took = self.end_time - self.start_time
        print(f"[{self.prompt}]: {time_took:.4f}s")
