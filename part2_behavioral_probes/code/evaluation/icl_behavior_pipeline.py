from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.codebook_utils import batch_call_llm, make_prompt, parse_answers
from evaluation.cot_behavior_pipeline import (
    fleiss_kappa_three,
    generic_label_codebook,
    labels_from_codebook,
    load_structured_codebook,
    read_labelled_split,
    score_predictions,
    shuffled_codebook,
    swapped_label_codebook,
)


@dataclass
class IclRunConfig:
    dataset: str
    split_file: Path
    codebook_file: Path
    example_file: Path
    model_name: str
    quantization: str = "4"
    limit: int = 0
    batch_size: int = 4
    max_new_tokens: int = 64
    output_dir: Path = Path("results/icl_pipeline")
    text_file: Optional[Path] = None


def load_fewshot_examples(example_file: Path) -> str:
    return example_file.read_text(encoding="utf-8").strip()


def relabel_icl_answers(examples: str, mapping: Dict[str, str]) -> str:
    """Apply a label remapping to `Answer: LABEL` lines in the few-shot block."""
    if not mapping:
        return examples

    def repl(match: re.Match[str]) -> str:
        prefix, label = match.group(1), match.group(2).strip()
        return f"{prefix}{mapping.get(label, label)}"

    return re.sub(r"(?im)^(Answer:\s*)([A-Za-z0-9_ -]+)\s*$", repl, examples)


def build_icl_user_message(document: str, examples: str, labels: Sequence[str]) -> str:
    labels_text = ", ".join(labels)
    return (
        "Classify the political relation between source (<S></S>) and target (<T></T>).\n\n"
        "Here are labeled examples:\n\n"
        f"{examples}\n\n"
        "Now classify:\n"
        f"Sentence: {document}\n\n"
        f"Labels: {labels_text}\n"
        "Output ONLY the label name, nothing else."
    )


def parse_icl_answers(raw_answers: Sequence[str], allowed_labels: Sequence[str]) -> List[Optional[str]]:
    parsed = parse_answers(raw_answers, list(allowed_labels))
    answer_lines = []
    for text in raw_answers:
        match = re.search(r"Answer:\s*([A-Za-z0-9_ -]+)", str(text), flags=re.IGNORECASE)
        answer_lines.append(match.group(1).strip() if match else str(text))
    fallback = parse_answers(answer_lines, list(allowed_labels))
    return [p if p is not None else f for p, f in zip(parsed, fallback)]


def make_icl_prompts(
    documents: Sequence[str],
    examples: str,
    labels: Sequence[str],
    tokenizer: Any,
    *,
    system_message: str = "You are a precise classifier. Follow output format exactly.",
) -> List[str]:
    return [
        make_prompt(tokenizer, system_message, build_icl_user_message(str(doc), examples, labels))
        for doc in documents
    ]


def predict_icl(
    documents: Sequence[str],
    examples: str,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    max_new_tokens: int,
    seed: int = 42,
) -> Tuple[List[Optional[str]], List[str]]:
    prompts = make_icl_prompts(documents, examples, labels, tokenizer)
    raw_answers = batch_call_llm(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return parse_icl_answers(raw_answers, labels), raw_answers


def run_task_eval(
    df: pd.DataFrame,
    examples: str,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: IclRunConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    predictions, raw_answers = predict_icl(
        df["text"].astype(str).tolist(),
        examples,
        labels,
        model,
        tokenizer,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    scored = df.copy()
    scored["prediction"] = predictions
    scored["prediction_raw"] = raw_answers
    metrics = score_predictions(scored["label"].astype(str).tolist(), predictions, labels)
    metrics["unknown_rate"] = float(np.mean([p is None for p in predictions]))
    return scored, metrics


def _result(probe: str, metric: str, value: float, config: IclRunConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "behavioral_probe": probe,
        "metric": metric,
        "value": value,
        "dataset": config.dataset,
        "model_type": config.model_name,
        "quantization": config.quantization,
        "limit": config.limit,
        "prompt_style": "icl",
        "codebook_file": str(config.codebook_file),
        "example_file": str(config.example_file),
    }
    if extra:
        row.update(extra)
    return row


def _predict(
    documents: Sequence[str],
    examples: str,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: IclRunConfig,
) -> Tuple[List[Optional[str]], List[str]]:
    return predict_icl(
        documents,
        examples,
        labels,
        model,
        tokenizer,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )


def run_icl_behavior_probes(
    df: Optional[pd.DataFrame],
    codebook_list: Sequence[Dict[str, str]],
    examples: str,
    model: Any,
    tokenizer: Any,
    config: IclRunConfig,
    *,
    run_document_probes: bool = True,
    run_codebook_probes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels_from_codebook(codebook_list)
    records: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    if not run_document_probes and not run_codebook_probes:
        raise ValueError("At least one of run_document_probes or run_codebook_probes must be True.")

    if run_document_probes:
        if df is None or df.empty:
            raise ValueError("Document probes need a non-empty split dataframe with text + label columns.")

        docs = df["text"].astype(str).tolist()
        gold = df["label"].astype(str).tolist()
        base_pred, base_raw = _predict(docs, examples, labels, model, tokenizer, config)
        for metric, value in score_predictions(gold, base_pred, labels).items():
            records.append(_result("icl_dev_baseline", metric, value, config))
        records.append(_result("icl_legal_predictions", "legal_rate", float(np.mean([p in labels for p in base_pred])), config))
        frames.append(pd.DataFrame({"probe": "baseline", "text": df["text"], "label": df["label"], "prediction": base_pred, "raw": base_raw}))

        order_predictions: List[List[Optional[str]]] = []
        for probe, variant in (
            ("icl_order_reversed", list(codebook_list)[::-1]),
            ("icl_order_shuffled", shuffled_codebook(codebook_list)),
        ):
            variant_labels = labels_from_codebook(variant)
            pred, raw = _predict(docs, examples, variant_labels, model, tokenizer, config)
            for metric, value in score_predictions(gold, pred, labels).items():
                records.append(_result(probe, metric, value, config))
            records.append(_result(probe, "percent_change_from_baseline", float(np.mean([a != b for a, b in zip(base_pred, pred)])), config))
            order_predictions.append(pred)
            frames.append(pd.DataFrame({"probe": probe, "text": df["text"], "label": df["label"], "prediction": pred, "raw": raw}))

        if len(order_predictions) == 2:
            records.append(_result(
                "icl_order_fleiss_kappa",
                "fleiss_kappa",
                fleiss_kappa_three(base_pred, order_predictions[0], order_predictions[1], labels),
                config,
            ))

        gen_codebook, gen_map = generic_label_codebook(codebook_list)
        gen_labels = labels_from_codebook(gen_codebook)
        gen_gold = [gen_map.get(str(x).strip(), "LABEL_NA") for x in gold]
        gen_examples = relabel_icl_answers(examples, gen_map)
        gen_pred, gen_raw = _predict(docs, gen_examples, gen_labels, model, tokenizer, config)
        for metric, value in score_predictions(gen_gold, gen_pred, gen_labels).items():
            records.append(_result("icl_generic_labels", metric, value, config))
        frames.append(pd.DataFrame({"probe": "generic_labels", "text": df["text"], "label": gen_gold, "prediction": gen_pred, "raw": gen_raw}))

        swap_codebook, swap_map = swapped_label_codebook(codebook_list)
        swap_labels = labels_from_codebook(swap_codebook)
        swap_gold = [swap_map.get(str(x).strip(), "LABEL_NA") for x in gold]
        swap_examples = relabel_icl_answers(examples, swap_map)
        swap_pred, swap_raw = _predict(docs, swap_examples, swap_labels, model, tokenizer, config)
        for metric, value in score_predictions(swap_gold, swap_pred, swap_labels).items():
            records.append(_result("icl_swapped_labels", metric, value, config))
        frames.append(pd.DataFrame({"probe": "swapped_labels", "text": df["text"], "label": swap_gold, "prediction": swap_pred, "raw": swap_raw}))

    if run_codebook_probes:
        definition_docs = [str(row.get("Definition", "")).strip() for row in codebook_list]
        if any(definition_docs):
            def_pred, def_raw = _predict(definition_docs, examples, labels, model, tokenizer, config)
            for metric, value in score_predictions(labels, def_pred, labels).items():
                records.append(_result("icl_definition_recovery", metric, value, config))
            records.append(_result("icl_definition_legal_predictions", "legal_rate", float(np.mean([p in labels for p in def_pred])), config))
            frames.append(pd.DataFrame({"probe": "definition_recovery", "text": definition_docs, "label": labels, "prediction": def_pred, "raw": def_raw}))

        pos_docs, pos_gold = [], []
        neg_docs, neg_gold = [], []
        for row in codebook_list:
            if row.get("Positive Example"):
                pos_docs.append(row["Positive Example"])
                pos_gold.append(row["Label"])
            if row.get("Negative Example"):
                neg_docs.append(row["Negative Example"])
                neg_gold.append(row["Label"])

        if pos_docs:
            pos_pred, pos_raw = _predict(pos_docs, examples, labels, model, tokenizer, config)
            for metric, value in score_predictions(pos_gold, pos_pred, labels).items():
                records.append(_result("icl_positive_examples", metric, value, config))
            frames.append(pd.DataFrame({"probe": "positive_examples", "text": pos_docs, "label": pos_gold, "prediction": pos_pred, "raw": pos_raw}))

        if neg_docs:
            neg_pred, neg_raw = _predict(neg_docs, examples, labels, model, tokenizer, config)
            records.append(_result("icl_negative_examples", "not_original_label_rate", float(np.mean([p != g for p, g in zip(neg_pred, neg_gold)])), config))
            frames.append(pd.DataFrame({"probe": "negative_examples", "text": neg_docs, "label": neg_gold, "prediction": neg_pred, "raw": neg_raw}))

    return pd.DataFrame(records), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_pipeline_outputs(
    task_predictions: pd.DataFrame,
    task_metrics: Dict[str, float],
    behavior_results: pd.DataFrame,
    behavior_predictions: pd.DataFrame,
    config: IclRunConfig,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_part = config.model_name.split("/")[-1]
    stem = f"{config.dataset}_{config.split_file.stem}_{config.example_file.stem}_{config.codebook_file.stem}_{model_part}_limit{config.limit}"
    paths = {
        "task_predictions": config.output_dir / f"{stem}_task_predictions.csv",
        "task_metrics": config.output_dir / f"{stem}_task_metrics.csv",
        "behavior_results": config.output_dir / f"{stem}_behavior_results.csv",
        "behavior_predictions": config.output_dir / f"{stem}_behavior_predictions.csv",
    }
    task_predictions.to_csv(paths["task_predictions"], index=False)
    pd.DataFrame([task_metrics]).to_csv(paths["task_metrics"], index=False)
    behavior_results.to_csv(paths["behavior_results"], index=False)
    behavior_predictions.to_csv(paths["behavior_predictions"], index=False)
    return paths
