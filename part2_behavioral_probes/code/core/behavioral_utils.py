import copy
import json
from tqdm.auto import tqdm
import jsonlines
import numpy as np
from typing import List
import random
import pandas as pd

from core.codebook_utils import *
from core.paths import RESULTS_DIR, DATASET_SPLITS_DIR, PREDICTIONS_DIR


def _safe_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value).replace("\x00", "")


def _write_prediction_jsonl(output_file, documents, labels, predictions, raw_answers, source_list, target_list):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for n, pred in enumerate(predictions):
            row = {
                "document": _safe_text(documents[n]),
                "label": _safe_text(labels[n]),
                "prediction": _safe_text(pred),
                "raw_prediction": _safe_text(raw_answers[n]),
                "source": _safe_text(source_list[n]),
                "target": _safe_text(target_list[n]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def get_most_common_label(train_df):
    return train_df['label'].value_counts().idxmax()

def modify_codebook_with_exclusion(codebook_list, most_common, exclusion_prompt):
    """
    Add an exclusion prompt to the most common label in the codebook.
    """
    mod_codebook_list = copy.deepcopy(codebook_list)
    for item in mod_codebook_list:
        if item['Label'] == most_common:
            item['Definition'] += exclusion_prompt
    return mod_codebook_list

def create_modified_documents(df, doc_mod, limit):
    """
    Modify documents to include the "document modification" text.
    limit > 0: subsample; limit <= 0: use all rows (matches save_original_predictions when limit<=0).
    """
    if limit and limit > 0:
        try:
            df = df.sample(limit, random_state=42)
        except ValueError:
            pass
    normal_docs = df['text'].tolist()
    modified_docs = [doc + doc_mod for doc in normal_docs]
    return normal_docs, modified_docs

def get_predictions(
    documents,
    codebook,
    instruction_dict,
    model,
    tokenizer,
    meta_list=None,
    context_list=None,
    source_list=None,
    target_list=None,
):
    if meta_list is None:
        meta_list = [None] * len(documents)
    if context_list is None:
        context_list = [None] * len(documents)
    if source_list is None:
        source_list = [None] * len(documents)
    if target_list is None:
        target_list = [None] * len(documents)
    prompt_list = []
    raw_answers = []
    for n, document in tqdm(enumerate(documents), total=len(documents)):
        system_message, user_message = make_generic_message(
            document,
            codebook,
            instruction_dict,
            meta=meta_list[n],
            context=context_list[n],
            source=source_list[n],
            target=target_list[n],
        )
        prompt = make_prompt(tokenizer, system_message, user_message)
        prompt_list.append(prompt)
    raw_answers = batch_call_llm(prompt_list, model, tokenizer, max_new_tokens=30)
    return raw_answers

def get_predictions_batch(
    documents,
    codebook,
    instruction_dict,
    model,
    tokenizer,
    batch_size=32,
    meta_list=None,
    context_list=None,
    source_list=None,
    target_list=None,
):
    if meta_list is None:
        meta_list = [None] * len(documents)
    if context_list is None:
        context_list = [None] * len(documents)
    if source_list is None:
        source_list = [None] * len(documents)
    if target_list is None:
        target_list = [None] * len(documents)
    all_raw_answers = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        prompts = [
            make_prompt(
                tokenizer,
                *make_generic_message(
                    doc,
                    codebook,
                    instruction_dict,
                    meta=meta_list[i + j],
                    context=context_list[i + j],
                    source=source_list[i + j],
                    target=target_list[i + j],
                ),
            )
            for j, doc in enumerate(batch)
        ]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_width = int(inputs["input_ids"].shape[1])
        _pad = pad_token_id_for_generate(tokenizer)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                pad_token_id=_pad,
                do_sample=False,
            )
        for row in outputs:
            trimmed = row[input_width:]
            all_raw_answers.append(
                tokenizer.decode(trimmed.tolist(), skip_special_tokens=True)
            )
    return all_raw_answers

def calculate_accuracy(predictions, labels, inverse=False):
    if inverse:
        acc = [i != j for i, j in zip(predictions, labels)]
    else:
        acc = [i == j for i, j in zip(predictions, labels)]
    return np.mean(acc)

def prepare_codebook(codebook_list, excluded_sections, reverse=False):
    """
    Format the codebook for use in a prompt, excluding specified sections
    and reversing the order if desired.

    Parameters
    ----------
    codebook_list: List of dictionaries
        The codebook to format
    excluded_sections: List of strings
        The sections to exclude from the codebook
    reverse: bool
        If True, reverse the order of the codebook
    """
    if reverse:
        codebook_list = codebook_list[::-1]
    category_list = []
    for section in codebook_list:
        sub_section = {k: v for k, v in section.items() if k not in excluded_sections}
        sub_section = '\n'.join([f"{k}: {v}" for k, v in sub_section.items()]).strip()
        category_list.append(sub_section)
    return '\n\n'.join(category_list).strip()

# First, save all predictions on the dev set
def save_original_predictions(model_name,
                              model,
                              tokenizer,
                              dataset,
                              quantization,
                              excluded_sections=None,
                              limit=200):
    """
    Get the predictions for the original codebook and save them to a file. 
    """
    codebook_list, instruction_dict, _  = load_codebook(dataset)
    #categories = prepare_codebook(codebook_list, excluded_sections)
    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    if limit > 0:
        print("Limiting to", limit)
        dev_df = dev_df.sample(limit, random_state=42)
    documents = dev_df['text'].tolist()
    meta_list = dev_df['meta'].tolist()
    context_list = dev_df['context'].tolist()
    source_list = dev_df['source'].tolist() if 'source' in dev_df.columns else [None] * len(documents)
    target_list = dev_df['target'].tolist() if 'target' in dev_df.columns else [None] * len(documents)
    labels_raw = dev_df['label'].tolist()
    labels_cleaned = labels_raw

    raw_answers = get_predictions(
        documents,
        codebook_list,
        instruction_dict,
        model,
        tokenizer,
        meta_list,
        context_list,
        source_list,
        target_list,
    )
    clean_answers = parse_answers(raw_answers, labels_cleaned)
    model_name_part = model_name.split("/")[-1]
    _write_prediction_jsonl(
        PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_dev.jsonl",
        documents,
        labels_cleaned,
        clean_answers,
        raw_answers,
        source_list,
        target_list,
    )
            

def derangement_shuffle(lst, random_state=None):
    """Shuffle a list, ensuring that no element is in its original position."""
    if random_state:
        random.seed(random_state)
    n = len(lst)
    result = lst.copy()
    
    for i in range(n - 1):
        j = random.randint(i + 1, n - 1)
        result[i], result[j] = result[j], result[i]
    
    if result[-1] == lst[-1]:
        result[-1], result[-2] = result[-2], result[-1]
    
    return result



def permute_codebook_and_save(model_name,
                                model,
                                tokenizer,
                                quantization,
                                dataset="ccc",
                                modification="reverse",
                                excluded_sections=None,
                                limit=200):
    """
    Shuffle the order of the codebook and generate predictions.

    Then, load the saved predictions from the original order and 
    calculate the percentage change in predictions.
    """
    codebook_list, instruction_dict, _  = load_codebook(dataset)
    model_name_part = model_name.split("/")[-1]
    if modification == "reverse":
        codebook_list = codebook_list[::-1]
        output_file = PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_reverse_dev.jsonl"
    elif modification == "shuffle":
        # generate a random integer to use as a seed
        random.seed(42)
        random.shuffle(codebook_list)
        output_file = PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_shuffle_dev.jsonl"
    else:
        raise ValueError("Invalid modification type. Must be 'reverse' or 'shuffle'")

    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    if limit > 0:
        dev_df = dev_df.sample(limit, random_state=42)
    documents = dev_df['text'].tolist()
    meta_list = dev_df['meta'].tolist()
    context_list = dev_df['context'].tolist()
    source_list = dev_df['source'].tolist() if 'source' in dev_df.columns else [None] * len(documents)
    target_list = dev_df['target'].tolist() if 'target' in dev_df.columns else [None] * len(documents)
    labels_raw = dev_df['label'].tolist()
    labels_cleaned = labels_raw

    print(f"Beginning run on {len(documents)} documents...")
    raw_answers = get_predictions(
        documents,
        codebook_list,
        instruction_dict,
        model,
        tokenizer,
        meta_list,
        context_list,
        source_list,
        target_list,
    )

    clean_answers = parse_answers(raw_answers, labels_cleaned)
    _write_prediction_jsonl(
        output_file,
        documents,
        labels_cleaned,
        clean_answers,
        raw_answers,
        source_list,
        target_list,
    )

def save_generic_label_predictions(model_name,
                            quantization,
                            model,
                            tokenizer,
                            dataset="ccc",
                            limit=200):
    """
    Replace original, informative labels with generic labels and save predictions.
    """
    codebook_list, instruction_dict, _  = load_codebook(dataset)
    model_name_part = model_name.split("/")[-1]
    
    instruction_dict['Output Reminder'] = "Write the name of the Label that fits best, with no other text. For example, 'Label: LABEL_1', 'Label: LABEL_2', etc."

    #[i['Label'] for i in codebook_list]
    # make a dictionary of the labels, mapping from original to new
    label_dict = {i['Label']: f"LABEL_{n+1}" for n, i in enumerate(codebook_list)}
    for i in codebook_list:
        if 'Label' in i.keys():
            i['Label'] = label_dict[i['Label']]

    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    if limit > 0:
        dev_df = dev_df.sample(limit, random_state=42)
    documents = dev_df['text'].tolist()
    meta_list = dev_df['meta'].tolist()
    context_list = dev_df['context'].tolist()
    source_list = dev_df['source'].tolist() if 'source' in dev_df.columns else [None] * len(documents)
    target_list = dev_df['target'].tolist() if 'target' in dev_df.columns else [None] * len(documents)
    labels_raw = dev_df['label'].tolist()
    labels_cleaned = []
    for i in labels_raw:
        try:
            labels_cleaned.append(label_dict[i])
        except KeyError:
            labels_cleaned.append("LABEL_NA")

    print(f"Beginning run on {len(documents)} documents...")
    raw_answers = get_predictions(
        documents,
        codebook_list,
        instruction_dict,
        model,
        tokenizer,
        meta_list,
        context_list,
        source_list,
        target_list,
    )

    parsed_answers = parse_answers(raw_answers, labels_cleaned)
    parsed_answers = [i if i in labels_cleaned else "NA" for i in parsed_answers]
    labels_cleaned = labels_cleaned[:len(parsed_answers)]
    #pred_distribution = dict(Counter(parsed_answers))
    #true_distribution = dict(Counter(labels_cleaned))

    output_file = PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_generic_label_dev.jsonl"
    clean_answers = parse_answers(raw_answers, labels_cleaned)
    _write_prediction_jsonl(
        output_file,
        documents,
        labels_cleaned,
        clean_answers,
        raw_answers,
        source_list,
        target_list,
    )
            

def save_swapped_label_predictions(model_name,
                            quantization,
                            model,
                            tokenizer,
                            dataset="ccc",
                            limit=200):
    """
    Swap the order of the labels (that is, each label gets assigned to a different definition) and save.
    """
    codebook_list, instruction_dict, _  = load_codebook(dataset) 

    label_list = [i['Label'] for i in codebook_list]
    label_list = derangement_shuffle(label_list, random_state=42)
    label_dict = {i['Label']: j for i, j in zip(codebook_list, label_list)}

    for i in codebook_list:
        if 'Label' in i.keys():
            i['Label'] = label_dict[i['Label']]
    
    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    if limit > 0:
        dev_df = dev_df.sample(limit, random_state=42)
    documents = dev_df['text'].tolist()
    meta_list = dev_df['meta'].tolist()
    context_list = dev_df['context'].tolist()
    source_list = dev_df['source'].tolist() if 'source' in dev_df.columns else [None] * len(documents)
    target_list = dev_df['target'].tolist() if 'target' in dev_df.columns else [None] * len(documents)
    labels_raw = dev_df['label'].tolist()
    labels_cleaned = []
    for i in labels_raw:
        try:
            labels_cleaned.append(label_dict[i])
        except KeyError:
            labels_cleaned.append("Label_NA")

    print(f"Beginning run on {len(documents)} documents...")
    raw_answers = get_predictions(
        documents,
        codebook_list,
        instruction_dict,
        model,
        tokenizer,
        meta_list,
        context_list,
        source_list,
        target_list,
    )
    
    parsed_answers = parse_answers(raw_answers, labels_cleaned)
    parsed_answers = [i if i in labels_cleaned else "NA" for i in parsed_answers]
    labels_cleaned = labels_cleaned[:len(parsed_answers)]

    model_name_part = model_name.split("/")[-1]
    output_file = PREDICTIONS_DIR / f"{model_name_part}_quant_{quantization}_{dataset}_{limit}_swapped_label_dev.jsonl"
    clean_answers = parse_answers(raw_answers, labels_cleaned)
    _write_prediction_jsonl(
        output_file,
        documents,
        labels_cleaned,
        clean_answers,
        raw_answers,
        source_list,
        target_list,
    )

def fleiss_kappa(annotator1: List[str], annotator2: List[str], annotator3: List[str]) -> float:
    """
    Calculate Fleiss' Kappa for three annotators.
    
    Parameters
    ----------
    annotator1: List of categorical answers from the first annotator
    annotator2: List of categorical answers from the second annotator
    annotator3: List of categorical answers from the third annotator

    Returns
    -------
    Fleiss' Kappa value

    Example
    -------
    annotator1 = ['A', 'A', 'B', 'B']
    annotator2 = ['A', 'B', 'B', 'B']
    annotator3 = ['A', 'B', 'C', 'B']

    result = fleiss_kappa(annotator1, annotator2, annotator3)
    print(f"Fleiss' Kappa: {result:.4f}")
    """
    # replace all empty strings with 'NA'
    annotator1 = ['NA' if i == None else i for i in annotator1]
    annotator2 = ['NA' if i == None else i for i in annotator2]
    annotator3 = ['NA' if i == None else i for i in annotator3]

    if len(annotator1) != len(annotator2) or len(annotator1) != len(annotator3):
        raise ValueError("All annotators must have the same number of ratings")
    
    # Convert categorical answers to a matrix
    categories = sorted(set(annotator1 + annotator2 + annotator3))
    category_dict = {category: i for i, category in enumerate(categories)}
    
    n = len(annotator1)  # Number of subjects
    k = len(categories)  # Number of categories
    m = 3  # Number of annotators
    
    # Create the rating matrix
    matrix = np.zeros((n, k))
    for i in range(n):
        for annotator in [annotator1, annotator2, annotator3]:
            j = category_dict[annotator[i]]
            matrix[i, j] += 1
    
    # Calculate P_i (proportion of agreeing pairs for the i-th subject)
    P_i = (np.sum(matrix ** 2, axis=1) - m) / (m * (m - 1))
    P = np.mean(P_i)
    
    # Calculate P_e (proportion of agreeing pairs by chance)
    P_j = np.sum(matrix, axis=0) / (n * m)
    P_e = np.sum(P_j ** 2)
    
    # Calculate Fleiss' Kappa
    kappa = (P - P_e) / (1 - P_e)
    
    return kappa
