"""Behavioral reliability probes for codebook-grounded event coding.

The default run matches the reviewer-facing paper diagnostics: legal-label
compliance, definition recovery, order perturbations, generic-label probes,
swapped-mapping probes, and original-condition accuracy.
"""
# CSV `behavioral_probe` labels are defined in `probe_names.py`.
import os
import click
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Optional
from tqdm.auto import tqdm
import numpy as np
import jsonlines
from sklearn.metrics import f1_score
import re
import random
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.codebook_utils import (
    batch_call_llm,
    call_llm,
    load_codebook,
    load_model,
    load_tokenizer,
    make_prompt,
    parse_answers,
    resolve_codebook_paths,
)
from core.behavioral_utils import prepare_codebook, get_predictions, \
    create_modified_documents, modify_codebook_with_exclusion, save_swapped_label_predictions, \
    save_generic_label_predictions, calculate_accuracy, fleiss_kappa, permute_codebook_and_save, \
    save_original_predictions
from core.paths import RESULTS_DIR, DATASET_SPLITS_DIR, PREDICTIONS_DIR, BEHAVIORAL_RESULTS
from core.probe_names import (
    CODEBOOK_ALIGNMENT,
    DEV_BASELINE_ACCURACY,
    DEV_BASELINE_F1,
    I_LEGAL_LABELS,
    II_DEFINITION_RECOVERY,
    IIIA_IN_CONTEXT_POSITIVE,
    IIIB_IN_CONTEXT_NEGATIVE,
    IVA_ORDER_FLEISS,
    IVB_ORDER_REVERSED,
    IVC_ORDER_SHUFFLED,
    V_EXCLUSION_ALL,
    V_EXCLUSION_NORMAL,
    V_EXCLUSION_NORMAL_MODIFIED_DOC,
    V_EXCLUSION_MODIFIED_CODEBOOK_NORMAL_DOC,
    V_EXCLUSION_MODIFIED_CODEBOOK_MODIFIED_DOC,
    VI_GENERIC_ACCURACY,
    VI_GENERIC_F1,
    ORIGINAL_CONDITION_ACCURACY,
    RULE_FOLLOWING_SCORE,
    VII_SWAPPED_ACCURACY,
    VII_SWAPPED_F1,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test V column names (fixed; used when building the per-row correctness frame).
_EXCLUSION_VARIANT_COLUMNS = (
    "Normal codebook, normal doc",
    "Normal codebook, modified doc",
    "Modified codebook, normal doc",
    "Modified codebook, modified doc",
)

# Order used when saving the final CSV for display / review.
_PROBE_ORDER = {
    I_LEGAL_LABELS: 1,
    II_DEFINITION_RECOVERY: 2,
    IIIA_IN_CONTEXT_POSITIVE: 3,
    IIIB_IN_CONTEXT_NEGATIVE: 4,
    IVA_ORDER_FLEISS: 5,
    IVB_ORDER_REVERSED: 6,
    IVC_ORDER_SHUFFLED: 7,
    V_EXCLUSION_ALL: 8,
    V_EXCLUSION_NORMAL: 9,
    V_EXCLUSION_NORMAL_MODIFIED_DOC: 10,
    V_EXCLUSION_MODIFIED_CODEBOOK_NORMAL_DOC: 11,
    V_EXCLUSION_MODIFIED_CODEBOOK_MODIFIED_DOC: 12,
    VI_GENERIC_ACCURACY: 13,
    VI_GENERIC_F1: 14,
    VII_SWAPPED_ACCURACY: 15,
    VII_SWAPPED_F1: 16,
    DEV_BASELINE_ACCURACY: 17,
    DEV_BASELINE_F1: 18,
}


def _csv_label_key(s):
    """Match train/dev labels across dtypes (e.g. 14 vs 14.0 from read_csv)."""
    if pd.isna(s):
        return ""
    t = str(s).strip().upper()
    m = re.fullmatch(r"([+-]?\d+)\.0+", t)
    if m:
        return m.group(1)
    return t


def _progress(msg: str) -> None:
    """Print + log."""
    print(f"[behavioral_tests] {msg}", flush=True)
    logger.info(msg)


def _report_codebook_paths(dataset: str) -> None:
    """Print the effective codebook paths used for this dataset."""
    new_path, old_path = resolve_codebook_paths(dataset)
    msg = f"codebook / {dataset}: new={new_path.name}"
    if old_path == new_path:
        msg += " | old=same file"
    elif old_path.is_file():
        msg += f" | old={old_path.name}"
    else:
        msg += f" | old=missing -> fallback to {new_path.name}"
    _progress(msg)


def _sort_results_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows into paper test order before writing the CSV."""
    if "behavioral_probe" not in df.columns:
        return df
    out = df.copy()
    out["_probe_order"] = out["behavioral_probe"].map(_PROBE_ORDER).fillna(999)
    return out.sort_values(
        by=["dataset", "_probe_order", "behavioral_probe", "metric"],
        kind="stable",
    ).drop(columns=["_probe_order"])


def _first_metric_value(
    df: pd.DataFrame,
    probe: str,
    metric: str | None = None,
) -> float:
    rows = df[df["behavioral_probe"] == probe]
    if metric is not None:
        rows = rows[rows["metric"] == metric]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["value"])


def _mean_complete(values: list[float]) -> float:
    if any(pd.isna(v) for v in values):
        return float("nan")
    return float(np.mean(values))


def _paper_probe_breakdown(results: List[Dict]) -> pd.DataFrame:
    """Create the probe-level table reported in the paper appendix."""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    group_cols = ["dataset", "model_type", "quantization", "limit"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        dataset, model_type, quantization, limit = keys
        legal = _first_metric_value(g, I_LEGAL_LABELS, "accuracy")
        definition = _first_metric_value(g, II_DEFINITION_RECOVERY, "accuracy")
        orig = _first_metric_value(g, DEV_BASELINE_ACCURACY, "accuracy")
        order_kappa = _first_metric_value(g, IVA_ORDER_FLEISS, "fleiss_kappa")
        reverse_accuracy = _first_metric_value(g, IVB_ORDER_REVERSED, "accuracy")
        reverse_change = _first_metric_value(g, IVB_ORDER_REVERSED, "percent_change_reverse")
        shuffle_accuracy = _first_metric_value(g, IVC_ORDER_SHUFFLED, "accuracy")
        shuffle_change = _first_metric_value(g, IVC_ORDER_SHUFFLED, "percent_change_shuffle")
        generic_f1 = _first_metric_value(g, VI_GENERIC_F1, "f1")
        swap_f1 = _first_metric_value(g, VII_SWAPPED_F1, "f1")
        cb_align = _mean_complete([legal, definition])
        rule_s = _mean_complete([order_kappa, generic_f1, swap_f1])
        rows.append(
            {
                "dataset": dataset,
                "model_type": model_type,
                "quantization": quantization,
                "limit": limit,
                "orig_acc": orig,
                "legal_label_compliance": legal,
                "definition_recovery": definition,
                "cb_align": cb_align,
                "reverse_acc": reverse_accuracy,
                "reverse_change": reverse_change,
                "shuffle_acc": shuffle_accuracy,
                "shuffle_change": shuffle_change,
                "order_kappa": order_kappa,
                "generic_f1": generic_f1,
                "swap_f1": swap_f1,
                "rule_s": rule_s,
            }
        )
    return pd.DataFrame(rows)


def _paper_summary_rows(results: List[Dict]) -> pd.DataFrame:
    """Compact summary matching the paper's Table 4 terminology."""
    breakdown = _paper_probe_breakdown(results)
    if breakdown.empty:
        return pd.DataFrame()
    rows = []
    for _, row in breakdown.iterrows():
        base = {
            "dataset": row["dataset"],
            "model_type": row["model_type"],
            "quantization": row["quantization"],
            "limit": row["limit"],
        }
        rows.extend(
            [
                {**base, "summary_metric": ORIGINAL_CONDITION_ACCURACY, "value": row["orig_acc"]},
                {**base, "summary_metric": CODEBOOK_ALIGNMENT, "value": row["cb_align"]},
                {**base, "summary_metric": RULE_FOLLOWING_SCORE, "value": row["rule_s"]},
            ]
        )
    return pd.DataFrame(rows)


# Testing purposes
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#quantization = "4"
#model = load_model(model_name, quantization=quantization)
#tokenizer = load_tokenizer(model_name)


### Define each test as a separate function ###

def check_definition_recovery(model_name,
                              quantization,
                              model, 
                              tokenizer, 
                              dataset="ccc",
                              excluded_sections=[],
                              reverse=False):
    """
    Given a verbatim definition from the codebook, can the model correctly predict the label?
    """
    logger.info(f"Running Definition Recovery test for {dataset}")
    codebook_list, _instruction_dict, _old_codebook  = load_codebook(dataset) 

    logger.info("Preparing codebook")
    categories = prepare_codebook(codebook_list, excluded_sections, reverse=reverse)

    label_list = [] 
    prompts = []
    for i in tqdm(codebook_list):
        label_list.append(i['Label'])
        definition = i['Definition']
        system_message = "Instructions: Please read the definitions below and the provided document. Then return the Label that best fits the provided document. Don't write any other text except the Label."
        user_message = f"""Categories:\n{categories}\n\nDocument: {definition}\n\nReminder: Do not write anything except the Label.---\n"""
        prompt = make_prompt(tokenizer, system_message, user_message)
        prompts.append(prompt)
    
    raw_answers = batch_call_llm(prompts, model, tokenizer, max_new_tokens=30, seed=42)

    parsed_answers = parse_answers(raw_answers, label_list)

    acc = np.mean([i == j for i, j in zip(parsed_answers, label_list)])
    logger.info(f"Accuracy calculated: {acc}")

    ## Not used for now--could be used to check relationship between codebook position
    ## and accuracy
    data_list = []
    for n, i in enumerate(codebook_list):
        answer_str = "(correct)"
        if label_list[n] != parsed_answers[n]:
            answer_str = raw_answers[n]
        data_list.append({"label": i['Label'], 
                          "answer": answer_str,
                          "position": n})
    
    return [{"metric": "accuracy",
            "value": acc,
            "behavioral_probe": II_DEFINITION_RECOVERY,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset}]

#results = check_definition_recovery(model_name, quantization, model, tokenizer, dataset="ccc", excluded_sections=[], reverse=False)

### Classify in-context positive/negative examples
def check_in_context_examples(model_name,
                              quantization,
                              model, 
                              tokenizer, 
                              dataset="ccc",
                              excluded_sections=None):
    """
    Can the model correctly classify the provided positive and negative examples from the codebook?
    """
    logger.info(f"Running In-Context Examples test for {dataset}")
    codebook_list, instruction_dict, _  = load_codebook(dataset) 
    if excluded_sections is None:
        excluded_sections = []

    categories = prepare_codebook(codebook_list, excluded_sections)

    prompt_list = []
    label_list = [] 
    for i in tqdm(codebook_list):
        if 'Positive Example' in i.keys():
            label_list.append(i['Label'])
            definition = i['Positive Example']
            system_message = "Instructions: Please read the definitions below and the provided document. Then return the Label that best fits the provided document. Don't write any other text except the Label."
            user_message = f"""Categories:\n{categories}\n\nDocument: {definition}\n\nReminder: Do not write anything except the Label.---\n"""
            prompt = make_prompt(tokenizer, system_message, user_message)
            prompt_list.append(prompt)
    if len(prompt_list) < len(codebook_list):
        logger.warning(f"Not all codebook entries have a positive example: {len(prompt_list)} vs {len(codebook_list)}")

    raw_answers = batch_call_llm(prompt_list, model, tokenizer, max_new_tokens=30, seed=42)
    parsed_answers = parse_answers(raw_answers, [i['Label'] for i in codebook_list])
    acc_pos = np.mean([i == j for i, j in zip(parsed_answers, label_list)])
    logger.info(f"Percentage of positive examples recovered: {acc_pos}")

    prompt_list = []
    label_list = [] 
    for i in tqdm(codebook_list):
        if 'Negative Example' in i.keys():
            label_list.append(i['Label'])
            definition = i['Negative Example']
            system_message = "Instructions: Please read the definitions below and the provided document. Then return the Label that best fits the provided document. Don't write any other text except the Label."
            user_message = f"""Categories:\n{categories}\n\nDocument: {definition}\n\nReminder: Do not write anything except the Label.---\n"""
            prompt = make_prompt(tokenizer, system_message, user_message)
            prompt_list.append(prompt)
    if len(prompt_list) < len(codebook_list):
        logger.warning(f"Not all codebook entries have a negative example: {len(prompt_list)} vs {len(codebook_list)}")
    
    raw_answers = batch_call_llm(prompt_list, model, tokenizer, max_new_tokens=30, seed=42)
    parsed_answers = parse_answers(raw_answers, [i['Label'] for i in codebook_list])
    acc_neg = np.mean([i != j for i, j in zip(parsed_answers, label_list)]) 
    logger.info(f"Percentage of negative examples recovered: {acc_neg}")
    
    return [{"metric": "accuracy",
            "value": acc_pos,
            "behavioral_probe": IIIA_IN_CONTEXT_POSITIVE,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            },
            {"metric": "accuracy",
            "value": acc_neg,
            "behavioral_probe": IIIB_IN_CONTEXT_NEGATIVE,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset}]

#results = check_in_context_examples(model_name, quantization, model, tokenizer, dataset="bfrs", excluded_sections=[])


def check_exclusion_criteria_pairs(model_name,
                                   quantization,
                                   model, 
                                   tokenizer, 
                                   dataset, 
                                   doc_mod="\nAnd we also support elephants.", 
                                   exclusion_prompt="\nIMPORTANT NOTE: This category *does not* apply if the document discusses an elephant. If it mentions an elephant, this category is not the correct category.", 
                                   limit=200):
    """
    The exclusion criteria test.

    This one is more confusing than others, and draws inspiration from Karpinska et al. (2024)'s proposed
    assessment technique. In short, we create four versions of a codebook-document pair:
    - Normal codebook, normal doc
      - Should be positive (We only look at a single label)
    - Modified codebook, normal doc. The modification says that if a certain phrase/concept is mentioned, the label is not correct.
      - Should be positive
    - Modified codebook, modified doc.
      - Should be *negative*
    - Normal codebook, modified doc. Checks if the model gets distracted by the new phrase.
      - Should be positive.

    By default, the phrase is "And we also support elephants." and the exclusion prompt is about elephants.

    We only use documents that have the most common label in the train set.
    """
    logger.info(f"Running Exclusion Criteria test for {dataset}")
    train_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_train.csv")
    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    train_keys = train_df["label"].map(_csv_label_key)
    train_nonempty = train_keys[train_keys != ""]
    if train_nonempty.empty:
        raise ValueError(
            f"Exclusion test for {dataset} could not determine the most common train label."
        )
    train_label_counts = train_nonempty.value_counts()

    codebook_list, instruction_dict, _ = load_codebook(dataset)
    codebook_label_map = {
        _csv_label_key(item["Label"]): str(item["Label"]).strip()
        for item in codebook_list
        if str(item.get("Label", "")).strip()
    }

    freq_leader = train_label_counts.index[0]
    most_common_key = None
    for key_candidate, _freq in train_label_counts.items():
        if key_candidate in codebook_label_map:
            most_common_key = key_candidate
            break
    if most_common_key is None:
        raise ValueError(
            f"Exclusion test for {dataset}: no train label appears in the codebook Label set. "
            f"Train top labels include {train_label_counts.head(10).index.tolist()} but codebook defines "
            f"{sorted(codebook_label_map.keys())}. Fix the structured codebook or align split labels."
        )
    if most_common_key != freq_leader:
        logger.warning(
            "Exclusion test / %s: train-majority label %r missing from codebook; using highest-frequency "
            "train label present in codebook: %r (paper-style exclusion still uses one operationalized "
            "category from the loaded codebook.)",
            dataset,
            freq_leader,
            most_common_key,
        )
    label_raw = codebook_label_map[most_common_key]
    pos_df = dev_df[dev_df["label"].map(_csv_label_key) == most_common_key]
    if pos_df.empty:
        raise ValueError(
            f"Exclusion test for {dataset} picked train-majority label {label_raw!r}, "
            "but no matching dev rows were found."
        )
    _progress(f"exclusion label / {dataset}: {label_raw}")

    # Add the exclusion prompt to the codebook
    mod_codebook_list = modify_codebook_with_exclusion(codebook_list, label_raw, exclusion_prompt)
    if not any(_csv_label_key(item["Label"]) == most_common_key for item in codebook_list):
        logger.warning(
            "Exclusion test: chosen label %r has no matching codebook Label; exclusion clause "
            "was not applied.",
            label_raw,
        )

    # Align meta/context with the same rows as the documents (create_modified_documents
    # samples internally when limit>0; we sample here and pass limit=0 to avoid mismatch).
    pos_used = pos_df
    if limit and limit > 0 and len(pos_df) > 0:
        try:
            pos_used = pos_df.sample(
                n=min(int(limit), len(pos_df)), random_state=42
            )
        except ValueError:
            pos_used = pos_df
    normal_docs, modified_docs = create_modified_documents(pos_used, doc_mod, 0)
    labels = [label_raw] * len(normal_docs)

    # Fixed probe names (do not derive from `==` / `is`): if no codebook row matched
    # `most_common`, the two codebooks are structurally identical and string-based labels
    # used to collide and overwrite columns.
    exclusion_variants = [
        ("Normal codebook, normal doc", codebook_list, normal_docs, False),
        ("Normal codebook, modified doc", codebook_list, modified_docs, False),
        ("Modified codebook, normal doc", mod_codebook_list, normal_docs, False),
        ("Modified codebook, modified doc", mod_codebook_list, modified_docs, True),
    ]
    correct_dict = {}
    for probe_name, codebook, docs, inverse in exclusion_variants:
        predictions = get_predictions(
            docs,
            codebook,
            instruction_dict,
            model,
            tokenizer,
            meta_list=pos_used["meta"].tolist(),
            context_list=pos_used["context"].tolist(),
            source_list=pos_used["source"].tolist() if "source" in pos_used.columns else [None] * len(docs),
            target_list=pos_used["target"].tolist() if "target" in pos_used.columns else [None] * len(docs),
        )
        cleaned_predictions = parse_answers(predictions, labels)
        if inverse:
            correct = [i != j for i, j in zip(cleaned_predictions, labels)]
        else:
            correct = [i == j for i, j in zip(cleaned_predictions, labels)]
        correct_dict[probe_name] = correct

    correct_df = pd.DataFrame(
        {c: correct_dict.get(c, []) for c in _EXCLUSION_VARIANT_COLUMNS}
    )
    correct_df["all_correct"] = correct_df.all(axis=1)
    accuracy = correct_df["all_correct"].mean()

    results = [
        {
            "metric": "accuracy",
            "value": accuracy,
            "behavioral_probe": V_EXCLUSION_ALL,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            "limit": limit,
        },
        {
            "metric": "accuracy",
            "value": correct_df[_EXCLUSION_VARIANT_COLUMNS[0]].mean(),
            "behavioral_probe": V_EXCLUSION_NORMAL,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            "limit": limit,
        },
        {
            "metric": "accuracy",
            "value": correct_df[_EXCLUSION_VARIANT_COLUMNS[1]].mean(),
            "behavioral_probe": V_EXCLUSION_NORMAL_MODIFIED_DOC,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            "limit": limit,
        },
        {
            "metric": "accuracy",
            "value": correct_df[_EXCLUSION_VARIANT_COLUMNS[2]].mean(),
            "behavioral_probe": V_EXCLUSION_MODIFIED_CODEBOOK_NORMAL_DOC,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            "limit": limit,
        },
        {
            "metric": "accuracy",
            "value": correct_df[_EXCLUSION_VARIANT_COLUMNS[3]].mean(),
            "behavioral_probe": V_EXCLUSION_MODIFIED_CODEBOOK_MODIFIED_DOC,
            "model_type": model_name,
            "quantization": quantization,
            "dataset": dataset,
            "limit": limit,
        },
    ]

    return results

#check_exclusion_criteria_pairs(model_name, quantization, model, tokenizer, dataset="bfrs", limit=25)


def baseline_accuracy(model_name, quantization, dataset, model, tokenizer, limit=200):
    logger.info(f"Running Baseline accuracy test for {dataset}")
    model_name_part = model_name.split("/")[-1]
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
        assert data
    except Exception as exc:
        logger.debug("Prediction cache miss (original baseline): %s", exc)
        print(f"Could not find original predictions for {model_name_part} on {dataset}.")
        print("Generating original predictions...")
        save_original_predictions(model_name=model_name, 
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  dataset=dataset, 
                                  quantization=quantization, limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
    df = pd.DataFrame(data)
    predictions = df['prediction'].tolist()
    predictions = ["NONE" if i == None else i for i in predictions]
    labels = df['label'].tolist()
    acc = calculate_accuracy(predictions, labels)

    f1 = f1_score(labels, predictions, average='weighted')
    return [{"metric": "accuracy",
            "value": acc,
             "behavioral_probe": DEV_BASELINE_ACCURACY,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             'limit': limit},
            {"metric": "f1",
             "value": f1,
             "behavioral_probe": DEV_BASELINE_F1,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit}]

#baseline_accuracy(model_name, quantization, "bfrs", model, tokenizer, limit=24)


def calculate_permutation_fleiss_kappa(model_name, quantization,model, tokenizer, dataset, limit=200):
    logger.info(f"Running Permutation Fleiss Kappa test for {dataset}")
    # load saved predictions
    model_name_part = model_name.split("/")[-1]
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached original order predictions")
        df = pd.DataFrame(data)
        true_labels = df['label'].tolist()
        original_predictions = df['prediction'].tolist()
    except Exception as exc:
        logger.debug("Prediction cache miss (fleiss original): %s", exc)
        logger.info(f"Could not find original predictions for {model_name_part} on {dataset}.")
        logger.info("Generating original predictions...")
        save_original_predictions(model_name=model_name, 
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  dataset=dataset, 
                                  quantization=quantization, limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
            df = pd.DataFrame(data)
            true_labels = df['label'].tolist()
            original_predictions = df['prediction'].tolist()
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_reverse_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached reverse predictions")
        df = pd.DataFrame(data)
        reverse_predictions = df['prediction'].tolist()
    except Exception as exc:
        logger.debug("Prediction cache miss (reverse): %s", exc)
        logger.info(f"Could not find reverse predictions for {model_name_part} on {dataset}.")
        logger.info("Generating reverse predictions...")
        permute_codebook_and_save(model_name, model, tokenizer, quantization, dataset=dataset, modification="reverse", limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_reverse_dev.jsonl") as reader:
            data = list(reader)
        df = pd.DataFrame(data)
        reverse_predictions = df['prediction'].tolist()

    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_shuffle_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached shuffle predictions")
        df = pd.DataFrame(data)
        shuffle_predictions = df['prediction'].tolist()
    except Exception as exc:
        logger.debug("Prediction cache miss (shuffle): %s", exc)
        logger.info(f"Could not find shuffle predictions for {model_name_part} on {dataset}.")
        logger.info("Generating shuffle predictions...")
        permute_codebook_and_save(model_name, model, tokenizer, quantization, dataset=dataset, modification="shuffle", limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_shuffle_dev.jsonl") as reader:
            data = list(reader)
        df = pd.DataFrame(data)
        shuffle_predictions = df['prediction'].tolist()

    fleiss_kappa_score = fleiss_kappa(original_predictions, reverse_predictions, shuffle_predictions)

    # also calculate the percentage change in predictions between the original and reverse
    change_percent_reverse = np.mean([i != j for i, j in zip(original_predictions, reverse_predictions)])
    change_percent_shuffle = np.mean([i != j for i, j in zip(original_predictions, shuffle_predictions)])
    original_accuracy = np.mean([i == j for i, j in zip(true_labels, original_predictions)])
    reverse_accuracy = np.mean([i == j for i, j in zip(true_labels, reverse_predictions)])
    shuffle_accuracy = np.mean([i == j for i, j in zip(true_labels, shuffle_predictions)])


    if fleiss_kappa_score < 0.01:
        agreement = "Poor agreement"
    elif fleiss_kappa_score < 0.21:
        agreement = "Slight agreement"
    elif fleiss_kappa_score < 0.41:
        agreement = "Fair agreement"
    elif fleiss_kappa_score < 0.61:
        agreement = "Moderate agreement"
    elif fleiss_kappa_score < 0.81:
        agreement = "Substantial agreement"
    else:
        agreement = "Almost perfect agreement"

    return [{"metric": "fleiss_kappa",
               "value": fleiss_kappa_score,
               "behavioral_probe": IVA_ORDER_FLEISS,
               "fleiss_kappa_interpretation": agreement,
               "dataset": dataset,
               "model_type": model_name,
               "quantization": quantization,
               "limit": limit},
            {"metric": "accuracy",
             "value": original_accuracy,
             "behavioral_probe": IVA_ORDER_FLEISS,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit},
            {"metric": "percent_change_reverse",
             "value": change_percent_reverse,
              "behavioral_probe": IVB_ORDER_REVERSED,
              "dataset": dataset,
              "model_type": model_name,
              "quantization": quantization,
              "limit": limit},
            {"metric": "accuracy",
             "value": reverse_accuracy,
             "behavioral_probe": IVB_ORDER_REVERSED,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit},
            {"metric": "percent_change_shuffle",
                "value": change_percent_shuffle,
                "behavioral_probe": IVC_ORDER_SHUFFLED,
                "dataset": dataset,
                "model_type": model_name,
                "quantization": quantization,
                "limit": limit},
            {"metric": "accuracy",
                "value": shuffle_accuracy,
                "behavioral_probe": IVC_ORDER_SHUFFLED,
                "dataset": dataset,
                "model_type": model_name,
                "quantization": quantization,
                "limit": limit}]


#calculate_permutation_fleiss_kappa(model_name, quantization, model, tokenizer, dataset="bfrs", limit=24)


def check_legal_predictions(model_name,
                            quantization,
                            model,
                            tokenizer,
                            dataset,
                            limit=200):
    logger.info(f"Running Legal Predictions test for {dataset}")
    codebook_list, instruction_dict, _  = load_codebook(dataset)

    model_name_part = model_name.split("/")[-1]
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached original predictions")
    except Exception as exc:
        logger.debug("Prediction cache miss (legal test original): %s", exc)
        logger.info(f"Could not find original predictions for {model_name_part} (limit = {limit}) on {dataset}.")
        logger.info("Generating original predictions...")
        save_original_predictions(model_name=model_name, 
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  dataset=dataset, 
                                  quantization=quantization, limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl") as reader:
            data = list(reader)
    df = pd.DataFrame(data)
    # get % predictions that are in the codebook
    codebook_labels = [i['Label'] for i in codebook_list]
    legal_predictions = df[df['prediction'].isin(codebook_labels)]
    legal_percent = len(legal_predictions) / len(df)

    return [{"metric": "accuracy",
            "value": legal_percent,
            "behavioral_probe": I_LEGAL_LABELS,
            "dataset": dataset,
            "model_type": model_name,
            "quantization": quantization,
            "limit": limit}]

#check_legal_predictions(model_name, quantization, model, tokenizer, "bfrs", limit=24)


## Invariance to label name changes (non-binary case)
## Change all labels to label1, label2, and check if the model can still predict them correctly
def check_generic_label_acc(model_name,
                            quantization,
                            model, 
                           tokenizer, 
                           dataset="ccc",
                           limit=200):
    logger.info(f"Running Generic Label test for {dataset}")
    model_name_part = model_name.split("/")[-1]
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_generic_label_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached generic label predictions")
    except Exception as exc:
        logger.debug("Prediction cache miss (generic label): %s", exc)
        print(f"Could not find generic label predictions for {model_name_part} on {dataset}.")
        print("Generating generic label predictions...")
        save_generic_label_predictions(model_name=model_name, quantization=quantization, model=model, tokenizer=tokenizer, dataset=dataset, limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_generic_label_dev.jsonl") as reader:
            data = list(reader)
    df = pd.DataFrame(data)
    predictions = df['prediction'].tolist()
    predictions = ["NONE" if i == None else i for i in predictions]
    labels = df['label'].tolist()
    acc = calculate_accuracy(predictions, labels)
    logger.info(f"Accuracy calculated: {acc}")

    f1 = f1_score(labels, predictions, average='weighted')

    return [{"metric": "accuracy",
            "value": acc,
             "behavioral_probe": VI_GENERIC_ACCURACY,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit},
            {"metric": "f1",
             "value": f1,
             "behavioral_probe": VI_GENERIC_F1,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit}]

#check_generic_label_acc(model_name=model_name, quantization=quantization, model=model, tokenizer=tokenizer, dataset="bfrs", limit=24)

def check_label_swap(model_name,
                     quantization,
                     model, 
                    tokenizer, 
                    dataset="ccc",
                    limit=200):
    """
    Check if the model can still predict the correct label if the labels are swapped.
    That is, the label "PROTEST" might get the definition for "RALLY".
    """
    random.seed(42)
    logger.info(f"Running Swapped Label test for {dataset}")
    model_name_part = model_name.split("/")[-1]
    try:
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_swapped_label_dev.jsonl") as reader:
            data = list(reader)
        assert data
        logger.info("Using cached swapped label predictions")
    except Exception as exc:
        logger.debug("Prediction cache miss (swapped label): %s", exc)
        logger.info(f"Could not find swapped label predictions for {model_name_part} on {dataset}.")
        logger.info("Generating swapped label predictions...")
        random.seed(42)
        save_swapped_label_predictions(model_name, quantization, model, tokenizer, dataset, limit=limit)
        with jsonlines.open(PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_swapped_label_dev.jsonl") as reader:
            data = list(reader)
    df = pd.DataFrame(data)
    predictions = df['prediction'].tolist()
    predictions = ["NONE" if i == None else i for i in predictions]
    labels = df['label'].tolist()
    acc = calculate_accuracy(predictions, labels)

    f1 = f1_score(labels, predictions, average='weighted')

    return [{"metric": "accuracy",
            "value": acc,
             "behavioral_probe": VII_SWAPPED_ACCURACY,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit},
            {"metric": "f1",
             "value": f1,
             "behavioral_probe": VII_SWAPPED_F1,
             "dataset": dataset,
             "model_type": model_name,
             "quantization": quantization,
             "limit": limit}]
 

#check_label_swap(model_name=model_name, quantization=quantization, model=model, tokenizer=tokenizer, dataset="bfrs", limit=24)

### end of test definitions ###


def save_results(results: List[Dict], dataset: str, model_name: str, 
                quantization: str, limit: int, output_dir: Path):
    """Save one long CSV plus paper-style reviewer summaries."""
    model_name_part = model_name.split('/')[-1]
    if isinstance(dataset, (list, tuple)):
        dataset = "_".join(dataset)
    output_path = output_dir / f"{model_name_part}_{quantization}_{limit}_{dataset}_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append = os.environ.get("BEHAVIOR_APPEND_BEHAVIORAL_CSV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("DATAVERSE_APPEND_BEHAVIORAL_CSV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if append and output_path.exists():
        df = pd.read_csv(output_path)
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        df = _sort_results_for_csv(df)
        df.to_csv(output_path, index=False)
    else:
        df = _sort_results_for_csv(pd.DataFrame(results))
        df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    breakdown = _paper_probe_breakdown(results)
    if not breakdown.empty:
        breakdown_path = output_dir / f"{model_name_part}_{quantization}_{limit}_{dataset}_paper_probe_breakdown.csv"
        breakdown.to_csv(breakdown_path, index=False)
        logger.info(f"Paper probe breakdown saved to {breakdown_path}")

    summary = _paper_summary_rows(results)
    if not summary.empty:
        summary_path = output_dir / f"{model_name_part}_{quantization}_{limit}_{dataset}_paper_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Paper summary saved to {summary_path}")


def run_codebook_tests(model_name,  quantization, model, tokenizer, datasets, limit):
    """Run paper probes that only require the codebook."""
    all_results = []
    for dataset in datasets:
        results = []
        logger.info(f"Running codebook tests for {dataset}")
        _progress(f"codebook / {dataset}: definition recovery")
        results.extend(check_definition_recovery(model_name, quantization, model, tokenizer, 
                                              dataset=dataset))
        _progress(f"codebook / {dataset}: done.")
        for row in results:
            row.setdefault("limit", limit)
        all_results.extend(results)
    return all_results

def run_unlabeled_tests(model_name, quantization, model, tokenizer, datasets, limit):
    """Run tests that require text but not labels."""
    all_results = []
    for dataset in datasets:
        results = []
        logger.info(f"Running unlabeled tests for {dataset}")
        _progress(f"unlabeled / {dataset}: permutation Fleiss kappa")
        results.extend(calculate_permutation_fleiss_kappa(model_name, quantization, model, 
                                                       tokenizer, dataset, limit=limit))
        _progress(f"unlabeled / {dataset}: legal predictions")
        results.extend(check_legal_predictions(model_name, quantization, model, tokenizer,
                                           dataset, limit=limit))
        _progress(f"unlabeled / {dataset}: done.")
        all_results.extend(results)
    return all_results

def run_labeled_tests(model_name, quantization, model, tokenizer, datasets, limit):
    """Run paper probes that require labeled source-target examples."""
    all_results = []
    for dataset in datasets:
        results = []
        logger.info(f"Running labeled tests for {dataset}")
        _progress(f"labeled / {dataset}: baseline accuracy")
        results.extend(baseline_accuracy(model_name, quantization, dataset, model, 
                                     tokenizer, limit=limit))
        _progress(f"labeled / {dataset}: generic label")
        results.extend(check_generic_label_acc(model_name, quantization, model, tokenizer,
                                           dataset=dataset, limit=limit))
        _progress(f"labeled / {dataset}: label swap")
        results.extend(check_label_swap(model_name, quantization, model, tokenizer,
                                    dataset=dataset, limit=limit))
        _progress(f"labeled / {dataset}: done.")
        all_results.extend(results)
    return all_results


@click.command()
@click.option('--model-name', required=True, help='Name or path of the model to test')
@click.option('--quantization', default='4', help='Quantization level (default: 4)')
@click.option('--datasets', required=True, multiple=True, help='One or more datasets to test on')
@click.option('--limit', default=200, help='Number of samples to test')
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=BEHAVIORAL_RESULTS,
    help='Directory to save behavioral CSVs (default: <project>/results/behavioral_results)',
)
@click.option('--codebook', is_flag=True, help='Run codebook-only tests')
@click.option('--unlabeled', is_flag=True, help='Run tests requiring only unlabeled text')
@click.option('--labeled', is_flag=True, help='Run tests requiring labeled text')
def run_tests(model_name: str, 
              quantization: str, 
              datasets: List[str], 
              limit: int, 
              output_dir: Path,
              codebook: bool, 
              unlabeled: bool, 
              labeled: bool):
    """Run paper behavioral probes. If no test type is specified, runs all paper probes."""
    logger.info(f"##### BEGINNING TEST #####")
    logger.info(f"Running behavioral tests for {model_name} on datasets {datasets}")
    _progress(f"loaded from {Path(__file__).resolve()}")



    # parse out datasets, splitting on spaces or commas
    datasets = re.split(r'[ ,]', ' '.join(datasets))
    datasets = [i.lower().strip() for i in datasets if i]
    for dataset in datasets:
        _report_codebook_paths(dataset)
    
    # If no flags are set, run all tests
    run_all = not any([codebook, unlabeled, labeled])
    

    try:
        torch.backends.cuda.max_split_size_mb = 128
    except (AttributeError, RuntimeError):
        pass
    _progress("Loading model")
    model = load_model(model_name, quantization=quantization)
    _progress("Loading tokenizer")
    tokenizer = load_tokenizer(model_name)
    _progress("Model and tokenizer ready. Starting probes.")
    
    results = []
    
    if codebook or run_all:
        logger.info("Running codebook tests...")
        _progress("Phase: codebook tests")
        results.extend(run_codebook_tests(model_name, quantization, model, tokenizer, datasets, limit))
        _progress("Phase complete: codebook tests")
        
    if unlabeled or run_all:
        logger.info("Running unlabeled tests...")
        _progress("Phase: unlabeled tests")
        results.extend(run_unlabeled_tests(model_name, quantization, model, tokenizer, datasets, limit))
        _progress("Phase complete: unlabeled tests")
        
    if labeled or run_all:
        logger.info("Running labeled tests...")
        _progress("Phase: labeled tests")
        results.extend(run_labeled_tests(model_name, quantization, model, tokenizer, datasets, limit))
        _progress("Phase complete: labeled tests")
    
    _progress("Saving results")
    save_results(results, datasets, model_name, quantization, limit, output_dir)
    _progress("All done.")
    logger.info("Testing completed successfully")


if __name__ == '__main__':
    run_tests()
    ## Run all tests (no flags)
    # python behavioral_tests.py --model-name "mistralai/Mistral-7B-Instruct-v0.2" --datasets bfrs ccc
    # 
    ##  Run only codebook tests
    # python behavioral_tests.py --model-name "mistralai/Mistral-7B-Instruct-v0.2" --datasets bfrs --codebook
    # 
    ## Run labeled and unlabeled tests
    # python behavioral_tests.py --model-name "mistralai/Mistral-7B-Instruct-v0.2" --datasets bfrs --labeled --unlabeled
    # 
    ## Run all types explicitly
    # python behavioral_tests.py --model-name "mistralai/Mistral-7B-Instruct-v0.2" --datasets bfrs --codebook --labeled --unlabeled
