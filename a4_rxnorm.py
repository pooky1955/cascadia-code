from config import SQLITE_PATH, NDC_MAPPING_PATH, RXNAV_PATH
from util import save_pickle
import pandas as pd
import sqlite3
from collections import namedtuple

resolving_terms = [
    'has_ingredient',
    'has_tradename'
]


def load_sql_con(filepath=SQLITE_PATH):
    con = sqlite3.connect(filepath)
    return con


def load_dfs(con):
    rxconso_query = 'select * from RXNCONSO'
    rxrel_query = 'select * from RXNREL'
    rxcui_df = pd.read_sql_query(rxconso_query, con)
    rxrel_df = pd.read_sql_query(rxrel_query, con)
    resolve_mapping_df = rxrel_df[rxrel_df['RELA'].apply(
        lambda x: x in resolving_terms)]
    return rxcui_df, resolve_mapping_df


def create_mappings(rxcui_df):
    rxcui_str_to_id = {row['STR']: row['RXCUI']
                       for _, row in rxcui_df.iterrows()}
    rxcui_id_to_str = {row['RXCUI']: row['STR']
                       for _, row in rxcui_df.iterrows()}
    return rxcui_str_to_id, rxcui_id_to_str


def rxcui_values(df, rxcui_id_to_str):
    return [(rxcui, rxcui_id_to_str[rxcui]) for rxcui in df['RXCUI2'] if rxcui in rxcui_id_to_str]


def create_resolver_dict(rxrel_df,rxcui_id_to_str):
    rxcui_deep_dict = {rxcui_1: rxcui_values(
        df,rxcui_id_to_str) for rxcui_1, df in rxrel_df.groupby(by='RXCUI1')}
    return rxcui_deep_dict

def create_ndc_to_rxcui(con):
    ndc_query = 'select * from NDC'
    df = pd.read_sql_query(ndc_query,con)
    ndc_mapping = {row['NDC']:row['RXCUI'] for _, row in df.iterrows()}
    return ndc_mapping




RxNav = namedtuple('RxNav','rx_df drug_targets rx_str_id rx_id_str rx_resolver')

if __name__ == "__main__":
    con = load_sql_con()
    rxcui_df, rxrel_df = load_dfs(con)
    drug_targets = rxcui_df['STR'].unique()
    rxcui_str_to_id, rxcui_id_to_str = create_mappings(rxcui_df)
    rxcui_resolver_dict = create_resolver_dict(rxrel_df,rxcui_id_to_str)

    rxnav = RxNav(rx_df=rxcui_df,drug_targets=drug_targets,rx_str_id=rxcui_str_to_id,rx_id_str=rxcui_id_to_str,rx_resolver=rxcui_resolver_dict)
    save_pickle(rxnav,RXNAV_PATH)
    print("Saved RXNav")

    ndc_mapping = create_ndc_to_rxcui(con)
    save_pickle(ndc_mapping,NDC_MAPPING_PATH)
    print("Saved NDC-RXCUI Mapping")
