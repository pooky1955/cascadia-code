import pandas as pd
import numpy as np
from config import MIMIC_NOTES_PATH, MIMIC_NOTES_PICKLE_OUTPUT_PATH, MIMIC_NOTES_PICKLE_SHUFFLED
np.random.seed(42)

df = pd.read_csv(MIMIC_NOTES_PATH)
df.to_pickle(MIMIC_NOTES_PICKLE_OUTPUT_PATH)
inds = np.arange(len(df))
np.random.shuffle(inds)
df.iloc[inds].to_pickle(MIMIC_NOTES_PICKLE_SHUFFLED)
print(f"Converted {MIMIC_NOTES_PATH} to {MIMIC_NOTES_PICKLE_OUTPUT_PATH}")