import pandas as pd
from config import OUTPUT_DIR, COLUMN_ANNOTATED, COLUMN_RAW, COLUMN_ID
from os.path import join as pjoin
df = pd.read_csv(pjoin(OUTPUT_DIR,"n2c2.csv"))

