import pandas as pd
from config import MEDDRA_CSV_PATH

def load_meddra_se_df():
    all_side_effects_path = "./data/gzs/meddra_all_se.tsv"
    colnames = ['drug_id','a','b','c','d','adv_str']
    drug_adv_df = pd.read_csv(all_side_effects_path,sep='\t',header=None,names=colnames)[['drug_id','adv_str']]
    return drug_adv_df

def load_meddra_drugs_df():
    drug_df = pd.read_csv("data/gzs/drug_names.tsv",header=None,names=["id","name"],sep='\t')
    return drug_df

def make_meddra_drug_adv_df(drug_df,drug_adv_df,save_path=MEDDRA_CSV_PATH):
    sider_id_to_name = {row['id']:row['name'] for _, row in drug_df.iterrows()}
    drug_adv_df['drug_name'] = [sider_id_to_name[sider_id] for sider_id in drug_adv_df['drug_id'].values]
    drug_adv_df.to_csv(save_path)

def make_medra_dicts(meddra_df):
    return {drug_name : drug_df for drug_name, drug_df in meddra_df.groupby('drug_name')}

if __name__ == "__main__":
    drug_df = load_meddra_drugs_df()
    drug_adv_df = load_meddra_se_df()
    make_meddra_drug_adv_df(drug_df,drug_adv_df)
    print("Saved csv :)")