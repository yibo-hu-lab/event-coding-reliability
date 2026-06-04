from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from core.codebook_utils import (
    batch_call_llm,
    load_model,
    load_tokenizer,
    make_prompt,
    parse_answers,
    parse_new_codebook_format,
)


PLOVER_ROOTCODES = [
    "AGREE", "CONSULT", "SUPPORT", "COOPERATE", "AID", "YIELD",
    "REQUEST", "ACCUSE", "REJECT", "THREATEN",
    "PROTEST", "SANCTION", "MOBILIZE", "COERCE", "ASSAULT",
]

PLOVER_ROOT2QUAD = {
    "AGREE": 1, "CONSULT": 1, "SUPPORT": 1,
    "COOPERATE": 2, "AID": 2, "YIELD": 2,
    "REQUEST": 3, "ACCUSE": 3, "REJECT": 3, "THREATEN": 3,
    "PROTEST": 4, "SANCTION": 4, "MOBILIZE": 4, "COERCE": 4, "ASSAULT": 4,
}
PLOVER_ROOT2BIN = {r: (1 if PLOVER_ROOT2QUAD[r] <= 2 else 2) for r in PLOVER_ROOTCODES}


@dataclass
class CotRunConfig:
    dataset: str
    split_file: Path
    codebook_file: Path
    model_name: str
    quantization: str = "4"
    limit: int = 0
    batch_size: int = 2
    max_new_tokens: int = 512
    output_dir: Path = Path("results/cot_pipeline")
    text_file: Optional[Path] = None


def resolve_path(root: Path, spec: str | Path | None) -> Optional[Path]:
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def load_structured_codebook(dataset: str, codebook_file: Path) -> Tuple[List[Dict[str, str]], Dict[str, str], str]:
    codebook_list, instruction_dict = parse_new_codebook_format(dataset, new_format_path=codebook_file)
    raw = codebook_file.read_text(encoding="utf-8")
    return codebook_list, instruction_dict, raw


def labels_from_codebook(codebook_list: Sequence[Dict[str, str]]) -> List[str]:
    return [str(row["Label"]).strip() for row in codebook_list if str(row.get("Label", "")).strip()]


def format_codebook(
    codebook_list: Sequence[Dict[str, str]],
    *,
    excluded_sections: Optional[Iterable[str]] = None,
    reverse: bool = False,
) -> str:
    excluded = set(excluded_sections or [])
    rows = list(codebook_list)
    if reverse:
        rows = rows[::-1]
    chunks = []
    for row in rows:
        kept = {k: v for k, v in row.items() if k not in excluded}
        chunks.append("\n".join(f"{k}: {v}" for k, v in kept.items()).strip())
    return "\n\n".join(chunks).strip()


def shuffled_codebook(codebook_list: Sequence[Dict[str, str]], seed: int = 42) -> List[Dict[str, str]]:
    rows = [dict(r) for r in codebook_list]
    random.Random(seed).shuffle(rows)
    return rows


def generic_label_codebook(codebook_list: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    mapping = {str(row["Label"]).strip(): f"LABEL_{i + 1}" for i, row in enumerate(codebook_list)}
    out = []
    for row in codebook_list:
        copy = dict(row)
        copy["Label"] = mapping[str(row["Label"]).strip()]
        out.append(copy)
    return out, mapping


def swapped_label_codebook(codebook_list: Sequence[Dict[str, str]], seed: int = 42) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    labels = [str(row["Label"]).strip() for row in codebook_list]
    shuffled = labels[:]
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(labels, shuffled)):
            break
    else:
        shuffled = labels[1:] + labels[:1]
    mapping = dict(zip(labels, shuffled))
    out = []
    for row in codebook_list:
        copy = dict(row)
        copy["Label"] = mapping[str(row["Label"]).strip()]
        out.append(copy)
    return out, mapping


def build_cot_user_message(document: str, categories: str, labels: Sequence[str], task_type: str) -> str:
    labels_text = ", ".join(labels)
    if task_type == "aw":
        reasoning_steps = (
            "1. Who is source, who is target?\n"
            "2. What is the main action?\n"
            "3. Is it cooperative or conflictual?\n"
        )
    else:
        reasoning_steps = (
            "1. Who is source, who is target?\n"
            "2. What is the main action?\n"
            "3. Verbal (statements/promises) or material (physical)?\n"
            "4. Cooperative or conflictual?\n"
            "5. Which label fits best?\n"
        )
    return (
        "You are a political event classifier.\n\n"
        f"LABEL DEFINITIONS:\n{categories}\n\n"
        f"Sentence: {document}\n\n"
        "Think step by step:\n"
        f"{reasoning_steps}\n"
        "After reasoning, write final answer as:\n"
        f"ANSWER: <label>\n\n"
        f"Allowed labels: {labels_text}"
    )


def parse_cot_answers(raw_answers: Sequence[str], allowed_labels: Sequence[str]) -> List[Optional[str]]:
    answer_line = []
    for text in raw_answers:
        m = re.search(r"ANSWER:\s*([A-Za-z0-9_ -]+)", str(text), flags=re.IGNORECASE)
        answer_line.append(m.group(1).strip() if m else str(text))
    parsed = parse_answers(answer_line, list(allowed_labels))
    fallback = parse_answers(raw_answers, list(allowed_labels))
    return [p if p is not None else f for p, f in zip(parsed, fallback)]


def make_cot_prompts(
    documents: Sequence[str],
    codebook_list: Sequence[Dict[str, str]],
    tokenizer: Any,
    *,
    task_type: str,
    system_message: str = "You are a helpful assistant. Follow the requested answer format.",
) -> Tuple[List[str], List[str]]:
    labels = labels_from_codebook(codebook_list)
    categories = format_codebook(codebook_list)
    prompts = [
        make_prompt(tokenizer, system_message, build_cot_user_message(str(doc), categories, labels, task_type))
        for doc in documents
    ]
    return prompts, labels


def predict_cot(
    documents: Sequence[str],
    codebook_list: Sequence[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    *,
    task_type: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int = 42,
) -> Tuple[List[Optional[str]], List[str], List[str]]:
    prompts, labels = make_cot_prompts(documents, codebook_list, tokenizer, task_type=task_type)
    raw_answers = batch_call_llm(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    parsed = parse_cot_answers(raw_answers, labels)
    return parsed, raw_answers, labels


def read_labelled_split(split_file: Path, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(split_file)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{split_file} must contain text and label columns; got {list(df.columns)}")
    if limit and limit > 0:
        df = df.sample(min(int(limit), len(df)), random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def pessimistic_unknowns(predictions: Sequence[Optional[str]], gold: Sequence[str], labels: Sequence[str]) -> List[str]:
    out = []
    label_list = list(labels)
    for pred, true in zip(predictions, gold):
        if pred in label_list:
            out.append(str(pred))
            continue
        replacement = next((label for label in label_list if label != true), label_list[0])
        out.append(replacement)
    return out


def score_predictions(gold: Sequence[str], predictions: Sequence[Optional[str]], labels: Sequence[str]) -> Dict[str, float]:
    gold_clean = [str(x).strip() for x in gold]
    pred_clean = pessimistic_unknowns(predictions, gold_clean, labels)
    out: Dict[str, float] = {
        "accuracy": float(np.mean([p == g for p, g in zip(pred_clean, gold_clean)])),
        "macro_f1": float(f1_score(gold_clean, pred_clean, average="macro", labels=list(labels), zero_division=0)),
        "weighted_f1": float(f1_score(gold_clean, pred_clean, average="weighted", labels=list(labels), zero_division=0)),
    }
    if set(labels) == set(PLOVER_ROOTCODES):
        yq_t = [PLOVER_ROOT2QUAD.get(g, 0) for g in gold_clean]
        yq_p = [PLOVER_ROOT2QUAD.get(p, 0) for p in pred_clean]
        yb_t = [PLOVER_ROOT2BIN.get(g, 0) for g in gold_clean]
        yb_p = [PLOVER_ROOT2BIN.get(p, 0) for p in pred_clean]
        out["binary_f1"] = float(f1_score(yb_t, yb_p, average="macro", zero_division=0))
        out["quad_f1"] = float(f1_score(yq_t, yq_p, average="macro", zero_division=0))
        out["root_f1"] = out["macro_f1"]
    return out


def fleiss_kappa_three(
    a: Sequence[Optional[str]],
    b: Sequence[Optional[str]],
    c: Sequence[Optional[str]],
    labels: Sequence[str],
) -> float:
    """Fleiss' kappa for three raters over the same items."""
    normalized_rows = [
        tuple("NONE" if pred is None else str(pred) for pred in row)
        for row in zip(a, b, c)
    ]
    observed = sorted({pred for row in normalized_rows for pred in row})
    label_list = sorted(set(str(label) for label in labels) | set(observed))
    if not label_list:
        return float("nan")
    matrix = []
    for row in normalized_rows:
        counts = [sum(pred == label for pred in row) for label in label_list]
        matrix.append(counts)
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return float("nan")
    n_items, n_labels = arr.shape
    n_raters = 3.0
    p_i = (np.sum(arr * arr, axis=1) - n_raters) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_i))
    p_j = np.sum(arr, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j * p_j))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else float("nan")
    return float((p_bar - p_e) / (1.0 - p_e))


def run_task_eval(
    df: pd.DataFrame,
    codebook_list: Sequence[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    *,
    task_type: str,
    batch_size: int,
    max_new_tokens: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    predictions, raw_answers, labels = predict_cot(
        df["text"].astype(str).tolist(),
        codebook_list,
        model,
        tokenizer,
        task_type=task_type,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    scored = df.copy()
    scored["prediction"] = predictions
    scored["prediction_raw"] = raw_answers
    metrics = score_predictions(scored["label"].astype(str).tolist(), predictions, labels)
    metrics["unknown_rate"] = float(np.mean([p is None for p in predictions]))
    return scored, metrics


def _result(probe: str, metric: str, value: float, config: CotRunConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "behavioral_probe": probe,
        "metric": metric,
        "value": value,
        "dataset": config.dataset,
        "model_type": config.model_name,
        "quantization": config.quantization,
        "limit": config.limit,
        "prompt_style": "cot",
        "codebook_file": str(config.codebook_file),
    }
    if extra:
        row.update(extra)
    return row


def run_cot_behavior_probes(
    df: Optional[pd.DataFrame],
    codebook_list: Sequence[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    config: CotRunConfig,
    *,
    task_type: str,
    run_document_probes: bool = True,
    run_codebook_probes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels_from_codebook(codebook_list)
    records: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    if not run_document_probes and not run_codebook_probes:
        raise ValueError("At least one of run_document_probes or run_codebook_probes must be True.")

    if run_document_probes:
        if df is None or df.empty:
            raise ValueError("Document probes need a non-empty split dataframe with text + label columns.")

        base_pred, base_raw, _ = predict_cot(
            df["text"].astype(str).tolist(),
            codebook_list,
            model,
            tokenizer,
            task_type=task_type,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
        )
        base_metrics = score_predictions(df["label"].astype(str).tolist(), base_pred, labels)
        for metric, value in base_metrics.items():
            records.append(_result("cot_dev_baseline", metric, value, config))
        records.append(_result("cot_legal_predictions", "legal_rate", float(np.mean([p in labels for p in base_pred])), config))
        prediction_frames.append(pd.DataFrame({"probe": "baseline", "text": df["text"], "label": df["label"], "prediction": base_pred, "raw": base_raw}))

        for probe, variant in (
            ("cot_order_reversed", list(codebook_list)[::-1]),
            ("cot_order_shuffled", shuffled_codebook(codebook_list)),
        ):
            pred, raw, _ = predict_cot(
                df["text"].astype(str).tolist(),
                variant,
                model,
                tokenizer,
                task_type=task_type,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
            )
            metrics = score_predictions(df["label"].astype(str).tolist(), pred, labels)
            for metric, value in metrics.items():
                records.append(_result(probe, metric, value, config))
            records.append(_result(probe, "percent_change_from_baseline", float(np.mean([a != b for a, b in zip(base_pred, pred)])), config))
            prediction_frames.append(pd.DataFrame({"probe": probe, "text": df["text"], "label": df["label"], "prediction": pred, "raw": raw}))

        reverse_frame = prediction_frames[-2] if len(prediction_frames) >= 3 else None
        shuffle_frame = prediction_frames[-1] if len(prediction_frames) >= 3 else None
        if reverse_frame is not None and shuffle_frame is not None:
            records.append(_result(
                "cot_order_fleiss_kappa",
                "fleiss_kappa",
                fleiss_kappa_three(
                    base_pred,
                    reverse_frame["prediction"].tolist(),
                    shuffle_frame["prediction"].tolist(),
                    labels,
                ),
                config,
            ))

        gen_codebook, gen_map = generic_label_codebook(codebook_list)
        gen_labels = labels_from_codebook(gen_codebook)
        gen_gold = [gen_map.get(str(x).strip(), "LABEL_NA") for x in df["label"].tolist()]
        gen_pred, gen_raw, _ = predict_cot(
            df["text"].astype(str).tolist(),
            gen_codebook,
            model,
            tokenizer,
            task_type=task_type,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
        )
        gen_metrics = score_predictions(gen_gold, gen_pred, gen_labels)
        for metric, value in gen_metrics.items():
            records.append(_result("cot_generic_labels", metric, value, config))
        prediction_frames.append(pd.DataFrame({"probe": "generic_labels", "text": df["text"], "label": gen_gold, "prediction": gen_pred, "raw": gen_raw}))

        swap_codebook, swap_map = swapped_label_codebook(codebook_list)
        swap_labels = labels_from_codebook(swap_codebook)
        swap_gold = [swap_map.get(str(x).strip(), "LABEL_NA") for x in df["label"].tolist()]
        swap_pred, swap_raw, _ = predict_cot(
            df["text"].astype(str).tolist(),
            swap_codebook,
            model,
            tokenizer,
            task_type=task_type,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
        )
        swap_metrics = score_predictions(swap_gold, swap_pred, swap_labels)
        for metric, value in swap_metrics.items():
            records.append(_result("cot_swapped_labels", metric, value, config))
        prediction_frames.append(pd.DataFrame({"probe": "swapped_labels", "text": df["text"], "label": swap_gold, "prediction": swap_pred, "raw": swap_raw}))

    if run_codebook_probes:
        definition_docs = [row.get("Definition", "") for row in codebook_list]
        definition_gold = labels
        def_pred = []
        if any(definition_docs):
            def_pred, def_raw, _ = predict_cot(
                definition_docs,
                codebook_list,
                model,
                tokenizer,
                task_type=task_type,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
            )
            def_metrics = score_predictions(definition_gold, def_pred, labels)
            for metric, value in def_metrics.items():
                records.append(_result("cot_definition_recovery", metric, value, config))
            records.append(_result("cot_definition_legal_predictions", "legal_rate", float(np.mean([p in labels for p in def_pred])), config))
            prediction_frames.append(pd.DataFrame({"probe": "definition_recovery", "text": definition_docs, "label": definition_gold, "prediction": def_pred, "raw": def_raw}))

            for probe, variant in (
                ("cot_definition_order_reversed", list(codebook_list)[::-1]),
                ("cot_definition_order_shuffled", shuffled_codebook(codebook_list)),
            ):
                pred, raw, _ = predict_cot(
                    definition_docs,
                    variant,
                    model,
                    tokenizer,
                    task_type=task_type,
                    batch_size=config.batch_size,
                    max_new_tokens=config.max_new_tokens,
                )
                metrics = score_predictions(definition_gold, pred, labels)
                for metric, value in metrics.items():
                    records.append(_result(probe, metric, value, config))
                records.append(_result(probe, "percent_change_from_definition_baseline", float(np.mean([a != b for a, b in zip(def_pred, pred)])), config))
                prediction_frames.append(pd.DataFrame({"probe": probe, "text": definition_docs, "label": definition_gold, "prediction": pred, "raw": raw}))

            gen_codebook, gen_map = generic_label_codebook(codebook_list)
            gen_labels = labels_from_codebook(gen_codebook)
            gen_definition_docs = [row.get("Definition", "") for row in gen_codebook]
            gen_gold = [gen_map.get(str(label).strip(), "LABEL_NA") for label in definition_gold]
            gen_pred, gen_raw, _ = predict_cot(
                gen_definition_docs,
                gen_codebook,
                model,
                tokenizer,
                task_type=task_type,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
            )
            gen_metrics = score_predictions(gen_gold, gen_pred, gen_labels)
            for metric, value in gen_metrics.items():
                records.append(_result("cot_definition_generic_labels", metric, value, config))
            records.append(_result("cot_definition_generic_legal_predictions", "legal_rate", float(np.mean([p in gen_labels for p in gen_pred])), config))
            prediction_frames.append(pd.DataFrame({"probe": "definition_generic_labels", "text": gen_definition_docs, "label": gen_gold, "prediction": gen_pred, "raw": gen_raw}))

            swap_codebook, swap_map = swapped_label_codebook(codebook_list)
            swap_labels = labels_from_codebook(swap_codebook)
            swap_definition_docs = [row.get("Definition", "") for row in swap_codebook]
            swap_gold = [swap_map.get(str(label).strip(), "LABEL_NA") for label in definition_gold]
            swap_pred, swap_raw, _ = predict_cot(
                swap_definition_docs,
                swap_codebook,
                model,
                tokenizer,
                task_type=task_type,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
            )
            swap_metrics = score_predictions(swap_gold, swap_pred, swap_labels)
            for metric, value in swap_metrics.items():
                records.append(_result("cot_definition_swapped_labels", metric, value, config))
            records.append(_result("cot_definition_swapped_legal_predictions", "legal_rate", float(np.mean([p in swap_labels for p in swap_pred])), config))
            prediction_frames.append(pd.DataFrame({"probe": "definition_swapped_labels", "text": swap_definition_docs, "label": swap_gold, "prediction": swap_pred, "raw": swap_raw}))

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
            pos_pred, pos_raw, _ = predict_cot(pos_docs, codebook_list, model, tokenizer, task_type=task_type, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
            pos_metrics = score_predictions(pos_gold, pos_pred, labels)
            for metric, value in pos_metrics.items():
                records.append(_result("cot_positive_examples", metric, value, config))
            prediction_frames.append(pd.DataFrame({"probe": "positive_examples", "text": pos_docs, "label": pos_gold, "prediction": pos_pred, "raw": pos_raw}))
        if neg_docs:
            neg_pred, neg_raw, _ = predict_cot(neg_docs, codebook_list, model, tokenizer, task_type=task_type, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
            records.append(_result("cot_negative_examples", "not_original_label_rate", float(np.mean([p != g for p, g in zip(neg_pred, neg_gold)])), config))
            prediction_frames.append(pd.DataFrame({"probe": "negative_examples", "text": neg_docs, "label": neg_gold, "prediction": neg_pred, "raw": neg_raw}))

    result_df = pd.DataFrame(records)
    pred_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return result_df, pred_df


def save_pipeline_outputs(
    task_predictions: pd.DataFrame,
    task_metrics: Dict[str, float],
    behavior_results: pd.DataFrame,
    behavior_predictions: pd.DataFrame,
    config: CotRunConfig,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_part = config.model_name.split("/")[-1]
    stem = f"{config.dataset}_{config.split_file.stem}_{config.codebook_file.stem}_{model_part}_limit{config.limit}"
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

