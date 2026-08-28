import pandas as pd 
import json, ast
from itertools import combinations
from tqdm import tqdm

def get_combination(drugs, order):
    return [sorted(list(subset)) for subset in combinations(drugs, order)]

def contains_drug_pair_old(big_list, target):
    target_set = set(target)
    return any(set(item) == target_set for item in big_list)



df = pd.read_csv('HODDI/dataset/HODDI_v1/dictionary/Drugbank_ID_SMILE_all_structure links.csv')
drugid2drugname = dict(zip(df["DrugBank ID"].values, df["Name"].values ))
all_drug_id_list = df["DrugBank ID"].values

df = pd.read_csv('HODDI/dataset/HODDI_v1/dictionary/SE_similarity_2014Q3_2024Q3.csv')
sideffectid2sideffectname = dict(zip(df["recommended_umls_cui_from_meddra"].values, df["recommended_SE_name"].values ))

maindf = pd.read_csv("HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/pos.csv")
maindf["DrugBankID"] = maindf["DrugBankID"].apply(ast.literal_eval)
maindf["DrugBankID_sorted"] = [sorted(dbl) for dbl in maindf["DrugBankID"].values]

drug_lists = maindf["DrugBankID"].values
side_effect_list = maindf["SE_above_0.9"].values


maindf["dl_len"] = [len(set(dbl)) for dbl in maindf["DrugBankID_sorted"].values]


for ORDER in range(2,11):
    fp = open(f"{ORDER}-list.txt", "w")
    currdf = maindf[maindf.dl_len == ORDER].copy()
    print(f"ORDER ({ORDER}): ", len(currdf))
    count = 0
    #flag =1
    for idx, cdl in tqdm(zip(currdf.index, currdf["DrugBankID_sorted"].values)):
        #print(cdl)
        flag =1
        for cc in cdl:
            if cc not in all_drug_id_list:
                flag = -1
                break

        if flag == 1:
            # searching lower (OREDR-1) combinations
            comins = get_combination(cdl, ORDER-1)
            for c in comins:
                if not contains_drug_pair_old(maindf["DrugBankID_sorted"].values, c):
                    flag = -1
                    break
        if flag == 1:
            count +=1
            print(idx, file=fp)
        #break
    print(f"ORDER ({ORDER}), along with subset of order ({ORDER-1}): ", count)
    fp.close()

