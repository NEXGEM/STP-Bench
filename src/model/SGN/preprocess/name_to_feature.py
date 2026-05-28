#!/usr/bin/env python3

import argparse
import os
import pickle
import sys

os.environ['TRANSFORMERS_CACHE'] = "."
os.environ['HF_HOME'] = "."

import torch
import transformers
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extra import EXTRA

torch.backends.cuda.matmul.allow_tf32 = True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LLM_MODEL_NAME = 'Intel/neural-chat-7b-v3-1'

_SYSTEM_INPUTS = [
    (
        "You are a biology scientist specialising in gene study. Your mission is to describe "
        "the functionality and phenotype of the gene provided by the GeneCards gene symbol "
        "from the user. Your descriptions need to be concise and contain keywords only."
    ),
    (
        "You are a biology scientist specialising in gene study. Your mission is to describe "
        "the functionality and phenotype of the gene. The descriptions need to be concise and "
        "contain keywords only, providing the GeneCards gene symbol and its summary as a helpful "
        "reference. Note that the reference most likely contains no information on the "
        "functionality and phenotype of the gene. You are encouraged to complement the missing "
        "information for the functionality and phenotype of the gene. Do not directly copy from "
        "the reference unless you think it is extremely necessary."
    ),
]


def main(data_dir: str, overwrite: bool = False) -> None:
    output_dir = os.path.join(data_dir, "name_feature", _LLM_MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        _LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(_LLM_MODEL_NAME)

    gene2name_path = os.path.join(_SCRIPT_DIR, "gene2name.pkl")
    with open(gene2name_path, "rb") as f:
        names = pickle.load(f)
    names.update(EXTRA)

    if not overwrite:
        names = {
            k: v for k, v in names.items()
            if "symbol" in v
            and not os.path.exists(os.path.join(output_dir, v["symbol"] + ".pkl"))
        }
    else:
        names = {k: v for k, v in names.items() if "symbol" in v}

    print(f"Generating name features for {len(names)} genes -> {output_dir}")

    with torch.no_grad():
        for name in tqdm(names):
            entry = names[name]
            if "symbol" not in entry:
                continue
            symbol = entry["symbol"]
            out_path = os.path.join(output_dir, f"{symbol}.pkl")
            if os.path.exists(out_path) and not overwrite:
                continue

            summary = entry.get("summary")
            if summary is not None:
                summary = '.'.join(summary.strip().split('.')[:-2]) + '.'

            llm_output = [[], []]
            print_once = True
            for idx in range(2):
                if idx == 0:
                    system_input = _SYSTEM_INPUTS[0]
                    user_input = symbol
                else:
                    if summary is None:
                        continue
                    system_input = _SYSTEM_INPUTS[1]
                    user_input = f"GeneCards gene symbol: {symbol}. Reference: {summary}"

                prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
                inputs = tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                outputs = model.generate(
                    inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=5,
                )
                for output in outputs:
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    response = response.split("### Assistant:\n")[-1].strip()
                    feature = tokenizer.encode(
                        response, return_tensors="pt", add_special_tokens=False
                    ).to(model.device)
                    feature = model(feature, output_hidden_states=True).hidden_states[-1]
                    llm_output[0].append(response)
                    llm_output[1].append(feature)
                    if print_once:
                        print(response, flush=True)
                        print_once = False

            with open(out_path, "wb") as f:
                pickle.dump(llm_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data dir; name features saved to {data_dir}/name_feature/")
    parser.add_argument("--overwrite", action="store_true", default=False)
    # Accept standard pipeline args this script doesn't use
    parser.add_argument("--asset_dir", type=str, default=None)
    parser.add_argument("--external_dir", type=str, default=None)
    parser.add_argument("--external_asset_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gene_type", type=str, default=None)
    parser.add_argument("--num_genes", type=int, default=None)
    parser.add_argument("--meta_dir", type=str, default=None)

    args = parser.parse_args()
    main(data_dir=args.data_dir, overwrite=args.overwrite)
