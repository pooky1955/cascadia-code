from os.path import expanduser
N2C_DIR = "data/n2c_training"
OUTPUT_DIR = "data/intermediate"
COLUMN_ID = "sample_id"
COLUMN_RAW = "raw"
COLUMN_ANNOTATED = "annotated"
MODEL_PATH = "abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2"
BATCH_SIZE = 1
ADVERSE_EVENT_TAG = "ADR"
DRUG_EVENT_TAG = "DRUG"
MAX_LENGTH = 512
SQLITE_PATH = expanduser("~/Documents/RxNorm_full_01032022/rxnorm.db")
MEDDRA_CSV_PATH = "data/intermediate/adverse_drugs.csv"
RXNAV_PATH = "rxnav.pickle"
NDC_MAPPING_PATH = "ndc_mapping.pickle"