import ast
import argparse
import pandas as pd 
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, login
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 

from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict


with open("../hftoken", "r") as fp:
    hftoken = fp.read()

#hftoken = 'ADD_TOKEN'
login(token=hftoken, add_to_git_credential=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Name or path of the model to use",
)
parser.add_argument(
    "--run_subset",
    type=int,
    default=500,
    help="Number of samples to run, instead of the entire dataset",
)

args = parser.parse_args()

model_name = args.model_name

# ============ LOAD MODEL ==============================
print(f"Using model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# ============ LOAD DATASET ==============================
indexes = pd.read_csv('../data/2-list.txt', names=['row_ids']).row_ids.values
#print(indexes)

maindf = pd.read_csv("../../HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/pos.csv")
maindf["DrugBankID"] = maindf["DrugBankID"].apply(ast.literal_eval)
maindf["DrugBankID_sorted"] = [sorted(dbl) for dbl in maindf["DrugBankID"].values]
drug_lists = maindf["DrugBankID"].values
side_effect_list = maindf["SE_above_0.9"].values

data = maindf.loc[indexes]
mask = [len(dd)==2 for dd in data.DrugBankID_sorted.values]
print("len=", len(data), "masked=", len(data[mask]))
data = data[mask]

df = pd.read_csv('../../HODDI/dataset/HODDI_v1/dictionary/Drugbank_ID_SMILE_all_structure links.csv')
drugid2drugname = dict(zip(df["DrugBank ID"].values, df["Name"].values ))

df = pd.read_csv('../../HODDI/dataset/HODDI_v1/dictionary/SE_similarity_2014Q3_2024Q3.csv')
sideffectid2sideffectname = dict(zip(df["recommended_umls_cui_from_meddra"].values, df["recommended_SE_name"].values ))


# pre-analysis =======================================
print("ALL drug combinations:",len(data), "SDE possibilities=",len(set(data['SE_above_0.9'].values)))

#=====================================================

STARTNUM = args.run_subset
data = data[:STARTNUM]
print("SUBSET drug combinations:",len(data), "SDE possibilities=",len(set(data['SE_above_0.9'].values)))
data_ades = data["SE_above_0.9"].values
data = data["DrugBankID"].values
ades = list(set(data_ades))

# ============ prepping ===============================
# ============================================================
# YES / NO TOKEN IDS
# ============================================================

yes_tokens = tokenizer.encode(
    " YES",
    add_special_tokens=False
)

no_tokens = tokenizer.encode(
    " NO",
    add_special_tokens=False
)

assert len(yes_tokens) == 1, \
    f"' YES' is not a single token: {yes_tokens}"

assert len(no_tokens) == 1, \
    f"' NO' is not a single token: {no_tokens}"

yes_id = yes_tokens[0]
no_id = no_tokens[0]

#=====================================================
log_file = f"../resultlogs/logger_logits_order_two_goldoftwo_againstall_{model_name.replace('/', '-')}.jsonl"


with open(log_file, "a") as f:

    for ii, (idx,dd,sde) in enumerate(tqdm(zip(indexes,data,data_ades))):

        #print(f"\n========== Example {ii} ==========", dd)

        # ----------------------------------------------------
        # Get drugs
        # ----------------------------------------------------
        #print(ii, idx, dd, sde)
        A, B = [
            drugid2drugname[drug_id]
            for drug_id in dd
        ]

        # Candidate ABC side effect
        ygold = sideffectid2sideffectname[sde]
        #print('==>', A,B,C, ygold)
        #'''testing
        for candidate_ade_id in ades:

            # ----------------------------------------------------
            # Prompt
            # ----------------------------------------------------
            candidate_ade = sideffectid2sideffectname[candidate_ade_id]
            prompt = f"""You are given a combination of drugs and a candidate adverse effect.

    Drug combination:
    {A}, {B}

    Candidate adverse effect:
    {candidate_ade}

    Is this adverse effect associated with this drug combination?

    Answer only YES or NO."""
            #print(prompt)
            # ----------------------------------------------------
            # Qwen chat template
            # ----------------------------------------------------

            messages = [
                {
                    "role": "system",
                    "content": "You are a clinical pharmacology assistant. Answer strictly based on pharmacological knowledge, not general caution or disclaimers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # inputs is a BatchEncoding in your setup
            inputs = inputs.to(model.device)

            # ----------------------------------------------------
            # Forward pass
            # ----------------------------------------------------

            with torch.inference_mode():
                outputs = model(**inputs)

            # ----------------------------------------------------
            # Logits for next token
            # ----------------------------------------------------

            logits = outputs.logits[:, -1, :]

            # ----------------------------------------------------
            # YES / NO logits
            # ----------------------------------------------------

            yes_logit_tensor = logits[0, yes_id]
            no_logit_tensor = logits[0, no_id]

            # ----------------------------------------------------
            # Normalize over YES / NO
            # ----------------------------------------------------

            yn_logits = torch.stack([
                yes_logit_tensor,
                no_logit_tensor
            ])

            probs = torch.softmax(
                yn_logits,
                dim=0
            )

            p_yes = probs[0].item()
            p_no = probs[1].item()

            # Convert tensors to Python floats
            yes_logit = yes_logit_tensor.item()
            no_logit = no_logit_tensor.item()

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------

            record = {
                "line_id": ii,

                "A": A,
                "B": B,

                "y": candidate_ade,
                "ygold": ygold,

                "prompt": prompt,

                "yes_logit": yes_logit,
                "no_logit": no_logit,

                "p_yes": p_yes,
                "p_no": p_no
            }

            f.write(
                json.dumps(record) + "\n"
            )

            f.flush()
            #'''