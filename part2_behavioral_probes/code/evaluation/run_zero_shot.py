# read in the new codebook, parse it into JSON
import argparse
import itertools
import sys
import time
from collections import Counter
from typing import List
from pathlib import Path

import jsonlines
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.codebook_utils import (
    batch_call_llm,
    load_codebook,
    load_model,
    load_tokenizer,
    make_generic_message,
    make_generic_message_old,
    make_prompt,
    parse_answers,
)
from core.paths import ZERO_SHOT_RESULTS, DATASET_SPLITS_DIR

def run_zero_shot(model,
                  tokenizer,
                  dataset="bfrs",
                  model_type="mistral",
                  codebook_type="new_format",
                  eval_n_per_class=None,
                  constrained_generation=1.0,
                  excluded_sections=None,
                  max_new_tokens=20,
                  task_order="codebook_first",
                  seed=42,
                  split="dev",
                  return_raw_docs=False):
    """
    """
    if excluded_sections is None:
        excluded_sections = []

    codebook_list, instruction_dict, old_codebook = load_codebook(dataset)

    if eval_n_per_class < 0:
        eval_n_per_class = None

    if split == "dev":
        dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    elif split == "test":
        print("WARNING: Using test (held-out) split for evaluation!")
        dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_test.csv")
    elif split == "eval":
        print("WARNING: `eval` is deprecated; using the held-out test split.")
        dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_test.csv")
    else:
        raise ValueError("Invalid split. Must be one of 'dev' or 'test'")
    if eval_n_per_class:
        print("NOTE!! Only evaluating on a subset of the data! This is for testing purposes only.")
        dev_df = dev_df.groupby('label').apply(lambda x: x.sample(min(eval_n_per_class, len(x)), random_state=seed)).reset_index(drop=True)
    documents = dev_df['text'].tolist()
    if 'meta' not in excluded_sections:
        meta_list = dev_df['meta'].tolist()
    else:
         # list of None
        meta_list = [None for i in range(len(documents))]
    if 'context' not in excluded_sections:
        context_list = dev_df['context'].tolist()
    else:
        context_list = [None for i in range(len(documents))]
    source_list = dev_df['source'].tolist() if 'source' in dev_df.columns else [None for _ in range(len(documents))]
    target_list = dev_df['target'].tolist() if 'target' in dev_df.columns else [None for _ in range(len(documents))]
    labels_cleaned = dev_df['label'].tolist()
    print(f"Beginning run on {len(documents)} documents...")
    raw_answers = []
    prompt_list = []
    for n, document in tqdm(enumerate(documents), total=len(documents)):
        if codebook_type == "original":
            system_message, user_message = make_generic_message_old(document, old_codebook, instruction_dict)
        elif codebook_type == "new_format":
            system_message, user_message = make_generic_message(document, 
                                                        codebook_list, 
                                                        instruction_dict,
                                                        meta=meta_list[n],
                                                        context=context_list[n],
                                                        source=source_list[n],
                                                        target=target_list[n],
                                                        task_order=task_order,
                                                        excluded_sections=excluded_sections)
        else:
            raise ValueError("Invalid codebook type. Must be one of 'original' or 'new_format'")
        prompt = make_prompt(tokenizer, system_message, user_message)
        prompt_list.append(prompt)

    raw_answers = batch_call_llm(prompt_list, model, tokenizer, max_new_tokens=max_new_tokens,
                          constrained_generation=constrained_generation)
    parsed_answers = parse_answers(raw_answers, labels_cleaned)
    parsed_answers = [i if i in labels_cleaned else "NA" for i in parsed_answers]
    prec, rec, f1, _ = precision_recall_fscore_support(labels_cleaned, 
                                                       parsed_answers, 
                                                       average='weighted')
    
    #print(classification_report(labels_cleaned, parsed_answers))
    today = time.strftime("%Y-%m-%d")
    excluded_sections_str = "_".join([str(i) for i in excluded_sections])

    cr_file = ZERO_SHOT_RESULTS / f"zero_shot_classification_report_{dataset}_{model_type}_{excluded_sections_str}_{eval_n_per_class}.txt"
    with open(cr_file, "w") as f:
        f.write("Dataset:" + dataset + "\n")
        f.write("Model:" + model_type + "\n")
        f.write("Codebook:" + codebook_type + "\n")
        f.write("Constrained Generation:" + str(constrained_generation) + "\n")
        f.write("Excluded Sections:" + str(excluded_sections) + "\n")
        f.write("Split:" + split + "\n\n")
        f.write(classification_report(labels_cleaned, parsed_answers))

    pred_distribution = dict(Counter(parsed_answers))
    true_distribution = dict(Counter(labels_cleaned))
    d = {"precision": prec, 
                "recall": rec, 
                "f1": f1, 
                "eval_n_per_class": eval_n_per_class,
                "eval_n_total": len(documents),
                "model_type": model_type,
                "codebook_type": codebook_type,
                "dataset": dataset,
                "constrained_generation": constrained_generation,
                "excluded_sections": excluded_sections,
                "raw_answers": raw_answers,
                "parsed_answers": parsed_answers,
                "true_labels": labels_cleaned,
                "task_order": task_order,
                "true_distribution": true_distribution,
                "pred_distribution": pred_distribution,
                "dataset_split_used": split,
                }
    if return_raw_docs:
        d['documents'] = documents
        d['context'] = context_list
        d['source'] = source_list
        d['target'] = target_list
        d['prompt'] = prompt_list
    return d



def run_zero_shot_experiments(model_name: str, 
                              datasets: str,
                              split: str="dev",
                              eval_n_per_class: int=-1,
                              run_ablation: bool=False,
                              codebook_type_list: List[str] | None = None,
                              excluded_sections_list: list | None = None):
    print(f"Running zero-shot experiments on {model_name} model..., split={split}, eval_n_per_class={eval_n_per_class}")
    if codebook_type_list is None:
        codebook_type_list = ["new_format"]
    if excluded_sections_list is None:
        excluded_sections_list = [[]]
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    model_type = model_name.split("/")[-1]

    if datasets == 'all':
        dataset_list = ["manifestos", "ccc", "bfrs", "plover"]
    else:
        dataset_list = [i.strip() for i in datasets.split(",") if i]
    order_list = ["codebook_first"]
    constrained_generation_list = [1.0]
    today = time.strftime("%Y-%m-%d")

    if run_ablation:
        # ['Definition', 'Clarification', Negative Clarification, 'Positive Example', 'Negative Example', 'Output Reminder']
        excluded_sections_list = [["Output Reminder"], 
                                  ["Positive Example", "Negative Example"], 
                                  ["Positive Example", "Negative Example", "Clarification", "Negative Clarification"],
                                    ["Positive Example", "Negative Example", "Clarification", "Negative Clarification", "Output Reminder"],
                                    ["Positive Example", "Negative Example", "Clarification", "Negative Clarification", "Definition", "Output Reminder"]]
        print("Running ablation. Only running on new codebook format.")
        codebook_type_list = ["new_format"]
    else:
        excluded_sections_list = [[]]


    combo_list = list(itertools.product(dataset_list, codebook_type_list, excluded_sections_list, constrained_generation_list, order_list))

    print(f"Running on {len(combo_list)} combinations of dataset, codebook, and excluded sections...")
    for dataset, codebook_type, excluded_sections, constrained_generation, task_order in tqdm(combo_list, desc="Running zero-shot..."):
        if codebook_type == "original" and excluded_sections:
            continue
        print(f"Running zero-shot on {model_type} model with {codebook_type} codebook and excluded sections {excluded_sections}...")
        start_time = time.time()
        d = run_zero_shot(model,
                          tokenizer,
                          dataset=dataset,
                          model_type=model_type,
                          codebook_type=codebook_type,
                          excluded_sections=excluded_sections,
                          task_order=task_order,
                          eval_n_per_class=eval_n_per_class,
                          constrained_generation=constrained_generation,
                          split=split)
        end_time = time.time()
        elapsed_time = end_time - start_time
        d['elapsed_seconds'] = elapsed_time
        d['today'] = today

        with jsonlines.open(ZERO_SHOT_RESULTS / f"all_datasets_{model_type}_ablation={run_ablation}_{eval_n_per_class}.jsonl", mode='a') as writer:
            writer.write(d)

    # now write to CSV
    with jsonlines.open(ZERO_SHOT_RESULTS / f"all_datasets_{model_type}_ablation={run_ablation}_{eval_n_per_class}.jsonl", "r") as f:
        data = list(f.iter())
    df = pd.DataFrame(data)
    csv_filename = ZERO_SHOT_RESULTS / f"all_datasets_{model_type}_ablation={run_ablation}_{eval_n_per_class}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results written to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot experiments")
    parser.add_argument("--model-name", type=str, help="Type of model to use")
    parser.add_argument("--datasets", type=str, default="all", help="Which datasets to use")
    parser.add_argument("--split", type=str, default="dev", help="Split: dev or test (held-out)")
    parser.add_argument("--limit", type=int, default=-1, help="Number of examples to evaluate per class")
    parser.add_argument("--codebook-type-list", type=str, default="new_format", help="Which codebook format(s) to use--comma separated ('new_format' or 'original')")
    parser.add_argument("--run-ablation", action="store_true", help="Run ablation experiment?")
    args = parser.parse_args()
    codebook_type_list = args.codebook_type_list.split(",")

    run_zero_shot_experiments(model_name=args.model_name, 
                              datasets=args.datasets,
                              split=args.split, 
                              eval_n_per_class=args.limit, 
                              run_ablation=args.run_ablation,
                              codebook_type_list=codebook_type_list)





