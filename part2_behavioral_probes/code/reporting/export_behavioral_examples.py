"""Export reviewer-friendly examples for the paper behavioral probes."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import jsonlines
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.behavioral_utils import (
    create_modified_documents,
    derangement_shuffle,
    fleiss_kappa,
    get_predictions,
    modify_codebook_with_exclusion,
    save_generic_label_predictions,
    save_original_predictions,
    save_swapped_label_predictions,
    permute_codebook_and_save,
)
from core.codebook_utils import (
    batch_call_llm,
    load_codebook,
    load_model,
    load_tokenizer,
    make_prompt,
    parse_answers,
)
from core.paths import DATASET_SPLITS_DIR, RESULTS_DIR, PREDICTIONS_DIR

_DATASET_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


def _dataset_slug_arg(value: str) -> str:
    d = value.strip().lower()
    if len(d) >= 64 or not _DATASET_SLUG_RE.fullmatch(d):
        raise argparse.ArgumentTypeError(
            "Invalid dataset slug: use lowercase letters/digits plus optional '-' or '_' "
            "(max 63 chars)."
        )
    return d


DOC_MOD = "\nAnd we also support elephants."
EXCLUSION_PROMPT = (
    "\nIMPORTANT NOTE: This category *does not* apply if the document discusses an elephant. "
    "If it mentions an elephant, this category is not the correct category."
)
TEXT_COLUMNS = [
    "document",
    "raw_prediction",
    "model_raw_output",
    "prediction",
    "model_predicted_label",
    "model_parsed_label",
    "original_text",
    "modified_text",
    "tested_text",
    "example_text",
    "source",
    "target",
    "meta",
    "context",
]


def _test_output_dir(output_dir: Path, test_name: str) -> Path:
    path = output_dir / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _csv_label_key(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    match = re.fullmatch(r"([+-]?\d+)\.0+", text)
    if match:
        return match.group(1)
    return text


def _read_jsonl(path: Path) -> pd.DataFrame:
    with jsonlines.open(path, "r") as reader:
        rows = list(reader.iter())
    return pd.DataFrame(rows)


def _preserve_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out


def _write_error_analysis_csv(
    output_dir: Path,
    filename: str,
    df: pd.DataFrame,
    test_name: str,
    condition: str | None = None,
) -> None:
    error_dir = output_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)
    rows = _preserve_text_columns(df.reset_index(drop=True))
    rows.insert(0, "test_name", test_name)
    if condition is not None:
        if "condition" in rows.columns:
            rows["condition"] = rows["condition"].fillna("").astype(str)
        else:
            rows.insert(1, "condition", condition)
    rows["error_type"] = ""
    rows["note"] = ""
    rows.to_csv(error_dir / filename, index=False)


def _prediction_path(model_name: str, quantization: str, dataset: str, limit: int, suffix: str = "") -> Path:
    model_part = model_name.split("/")[-1]
    return PREDICTIONS_DIR / f"{model_part}_quant_{quantization}_{dataset}_{limit}{suffix}_dev.jsonl"


def _load_prediction_cache(
    model_name: str,
    quantization: str,
    model,
    tokenizer,
    dataset: str,
    limit: int,
    kind: str,
) -> pd.DataFrame:
    suffix_map = {
        "original": "",
        "reverse": "_reverse",
        "shuffle": "_shuffle",
        "generic_label": "_generic_label",
        "swapped_label": "_swapped_label",
    }
    path = _prediction_path(model_name, quantization, dataset, limit, suffix_map[kind])
    if path.is_file():
        return _read_jsonl(path)

    if kind == "original":
        save_original_predictions(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            quantization=quantization,
            limit=limit,
        )
    elif kind in {"reverse", "shuffle"}:
        permute_codebook_and_save(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            dataset=dataset,
            modification=kind,
            limit=limit,
        )
    elif kind == "generic_label":
        save_generic_label_predictions(
            model_name=model_name,
            quantization=quantization,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            limit=limit,
        )
    elif kind == "swapped_label":
        save_swapped_label_predictions(
            model_name=model_name,
            quantization=quantization,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            limit=limit,
        )
    else:
        raise ValueError(f"Unknown prediction kind: {kind}")

    return _read_jsonl(path)


def _dev_df(dataset: str, limit: int) -> pd.DataFrame:
    df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    if limit and limit > 0:
        df = df.sample(n=min(int(limit), len(df)), random_state=42)
    return df.reset_index(drop=True)


def export_test_i(
    output_dir: Path,
    dataset: str,
    original_df: pd.DataFrame,
    codebook_list: list[dict],
) -> None:
    test_dir = _test_output_dir(output_dir, "test_i")
    legal_labels = {str(item["Label"]).strip() for item in codebook_list if str(item.get("Label", "")).strip()}
    rows = _preserve_text_columns(original_df)
    rows["is_legal_prediction"] = rows["prediction"].isin(legal_labels)
    cols = [c for c in ["document", "label", "prediction", "raw_prediction", "is_legal_prediction", "source", "target"] if c in rows.columns]
    rows[cols].to_csv(test_dir / f"{dataset}_test_i_legal_examples.csv", index=False)


def export_test_ii(output_dir: Path, dataset: str, model, tokenizer, codebook_list: list[dict]) -> None:
    test_dir = _test_output_dir(output_dir, "test_ii")
    categories = "\n\n".join(
        "\n".join(f"{k}: {v}" for k, v in section.items())
        for section in codebook_list
    ).strip()
    prompts = []
    labels = []
    definitions = []
    for item in codebook_list:
        label = item["Label"]
        definition = item["Definition"]
        system_message = (
            "Instructions: Please read the definitions below and the provided document. "
            "Then return the Label that best fits the provided document. Don't write any other text except the Label."
        )
        user_message = f"Categories:\n{categories}\n\nDocument: {definition}\n\nReminder: Do not write anything except the Label.---\n"
        prompts.append(make_prompt(tokenizer, system_message, user_message))
        labels.append(label)
        definitions.append(definition)
    raw = batch_call_llm(prompts, model, tokenizer, max_new_tokens=30, seed=42)
    parsed = parse_answers(raw, labels)
    _preserve_text_columns(pd.DataFrame(
        {
            "label": labels,
            "definition": definitions,
            "model_raw_output": raw,
            "model_parsed_label": parsed,
            "correct": [a == b for a, b in zip(parsed, labels)],
        }
    )).to_csv(test_dir / f"{dataset}_test_ii_definition_recovery_examples.csv", index=False)


def export_test_iii(output_dir: Path, dataset: str, model, tokenizer, codebook_list: list[dict]) -> None:
    test_dir = _test_output_dir(output_dir, "test_iii")
    categories = "\n\n".join(
        "\n".join(f"{k}: {v}" for k, v in section.items())
        for section in codebook_list
    ).strip()
    rows = []
    for example_type in ("Positive Example", "Negative Example"):
        prompts = []
        expected_labels = []
        texts = []
        example_labels = []
        for item in codebook_list:
            if example_type not in item:
                continue
            system_message = (
                "Instructions: Please read the definitions below and the provided document. "
                "Then return the Label that best fits the provided document. Don't write any other text except the Label."
            )
            example_text = item[example_type]
            user_message = f"Categories:\n{categories}\n\nDocument: {example_text}\n\nReminder: Do not write anything except the Label.---\n"
            prompts.append(make_prompt(tokenizer, system_message, user_message))
            expected_labels.append(item["Label"])
            example_labels.append(item["Label"])
            texts.append(example_text)
        raw = batch_call_llm(prompts, model, tokenizer, max_new_tokens=30, seed=42)
        parsed = parse_answers(raw, [item["Label"] for item in codebook_list])
        for label, text, raw_answer, parsed_answer in zip(example_labels, texts, raw, parsed):
            correct = parsed_answer == label if example_type == "Positive Example" else parsed_answer != label
            rows.append(
                {
                    "example_type": "positive" if example_type == "Positive Example" else "negative",
                    "label": label,
                    "example_text": text,
                    "model_raw_output": raw_answer,
                    "model_parsed_label": parsed_answer,
                    "correct": correct,
                }
            )
    _preserve_text_columns(pd.DataFrame(rows)).to_csv(test_dir / f"{dataset}_test_iii_in_context_examples.csv", index=False)


def export_test_iv(
    output_dir: Path,
    dataset: str,
    original_df: pd.DataFrame,
    reverse_df: pd.DataFrame,
    shuffle_df: pd.DataFrame,
) -> None:
    test_dir = _test_output_dir(output_dir, "test_iv")
    merged = pd.DataFrame({"document": original_df["document"], "label": original_df["label"]})
    for col in ("meta", "context", "source", "target"):
        if col in original_df.columns:
            merged[col] = original_df[col]
    merged["original_prediction"] = original_df["prediction"]
    if "raw_prediction" in original_df.columns:
        merged["original_raw_output"] = original_df["raw_prediction"]
    merged["reverse_prediction"] = reverse_df["prediction"]
    if "raw_prediction" in reverse_df.columns:
        merged["reverse_raw_output"] = reverse_df["raw_prediction"]
    merged["shuffle_prediction"] = shuffle_df["prediction"]
    if "raw_prediction" in shuffle_df.columns:
        merged["shuffle_raw_output"] = shuffle_df["raw_prediction"]
    merged["original_correct"] = merged["original_prediction"] == merged["label"]
    merged["reverse_correct"] = merged["reverse_prediction"] == merged["label"]
    merged["shuffle_correct"] = merged["shuffle_prediction"] == merged["label"]
    merged["changed_in_reverse"] = merged["original_prediction"] != merged["reverse_prediction"]
    merged["changed_in_shuffle"] = merged["original_prediction"] != merged["shuffle_prediction"]
    _preserve_text_columns(merged).to_csv(test_dir / f"{dataset}_test_iv_order_examples.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "fleiss_kappa",
                "value": fleiss_kappa(
                    merged["original_prediction"].tolist(),
                    merged["reverse_prediction"].tolist(),
                    merged["shuffle_prediction"].tolist(),
                ),
            },
            {"metric": "original_accuracy", "value": float(merged["original_correct"].mean())},
            {"metric": "reverse_accuracy", "value": float(merged["reverse_correct"].mean())},
            {"metric": "shuffle_accuracy", "value": float(merged["shuffle_correct"].mean())},
            {"metric": "percent_change_reverse", "value": float(merged["changed_in_reverse"].mean())},
            {"metric": "percent_change_shuffle", "value": float(merged["changed_in_shuffle"].mean())},
        ]
    )
    summary.to_csv(test_dir / f"{dataset}_test_iv_order_summary.csv", index=False)
    reverse_error_rows = merged[(~merged["reverse_correct"]) | (merged["changed_in_reverse"])].copy()
    shuffle_error_rows = merged[(~merged["shuffle_correct"]) | (merged["changed_in_shuffle"])].copy()
    _write_error_analysis_csv(
        test_dir,
        f"{dataset}_test_iv_reverse_error_analysis.csv",
        reverse_error_rows,
        "Test IV",
        "reverse",
    )
    _write_error_analysis_csv(
        test_dir,
        f"{dataset}_test_iv_shuffle_error_analysis.csv",
        shuffle_error_rows,
        "Test IV",
        "shuffle",
    )


def export_test_v(
    output_dir: Path,
    dataset: str,
    model,
    tokenizer,
    codebook_list: list[dict],
    instruction_dict: dict,
    limit: int,
) -> None:
    test_dir = _test_output_dir(output_dir, "test_v")
    train_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_train.csv")
    dev_df = pd.read_csv(DATASET_SPLITS_DIR / f"{dataset}_dev.csv")
    train_keys = train_df["label"].map(_csv_label_key)
    most_common_key = train_keys[train_keys != ""].value_counts().idxmax()
    label_map = {
        _csv_label_key(item["Label"]): str(item["Label"]).strip()
        for item in codebook_list
        if str(item.get("Label", "")).strip()
    }
    label_raw = label_map[most_common_key]
    pos_df = dev_df[dev_df["label"].map(_csv_label_key) == most_common_key].copy()
    if limit and limit > 0:
        pos_df = pos_df.sample(n=min(int(limit), len(pos_df)), random_state=42)
    pos_df = pos_df.reset_index(drop=True)

    normal_docs, modified_docs = create_modified_documents(pos_df, DOC_MOD, 0)
    mod_codebook_list = modify_codebook_with_exclusion(codebook_list, label_raw, EXCLUSION_PROMPT)
    labels = [label_raw] * len(normal_docs)
    meta_list = pos_df["meta"].tolist()
    context_list = pos_df["context"].tolist()
    source_list = pos_df["source"].tolist() if "source" in pos_df.columns else [None] * len(normal_docs)
    target_list = pos_df["target"].tolist() if "target" in pos_df.columns else [None] * len(normal_docs)

    variants = [
        ("normal_codebook_normal_doc", codebook_list, normal_docs, False),
        ("normal_codebook_modified_doc", codebook_list, modified_docs, False),
        ("modified_codebook_normal_doc", mod_codebook_list, normal_docs, False),
        ("modified_codebook_modified_doc", mod_codebook_list, modified_docs, True),
    ]
    slide_rows = []
    summary_rows = []
    for name, codebook, docs, inverse in variants:
        raw_predictions = get_predictions(
            docs,
            codebook,
            instruction_dict,
            model,
            tokenizer,
            meta_list=meta_list,
            context_list=context_list,
            source_list=source_list,
            target_list=target_list,
        )
        parsed_predictions = parse_answers(raw_predictions, labels)
        if inverse:
            correct = [pred != gold for pred, gold in zip(parsed_predictions, labels)]
            should_happen = f"The model should NOT predict {label_raw}."
        else:
            correct = [pred == gold for pred, gold in zip(parsed_predictions, labels)]
            should_happen = f"The model should still predict {label_raw}."
        if name == "normal_codebook_normal_doc":
            should_happen = f"The model should predict {label_raw}."
        summary_rows.append({"condition": name, "accuracy": float(pd.Series(correct).mean()) if correct else 0.0})
        for idx, (raw_pred, parsed_pred, is_correct) in enumerate(zip(raw_predictions, parsed_predictions, correct)):
            if inverse:
                failure_pattern = "predicted_excluded_label" if not is_correct else "correctly_avoided_excluded_label"
            else:
                failure_pattern = "failed_to_keep_target_label" if not is_correct else "kept_target_label"
            slide_rows.append(
                {
                    "sample_id": idx,
                    "target_label": label_raw,
                    "condition": name,
                    "what_should_happen": should_happen,
                    "original_text": pos_df.loc[idx, "text"],
                    "modified_text": modified_docs[idx],
                    "tested_text": docs[idx],
                    "model_raw_output": raw_pred,
                    "model_parsed_label": parsed_pred,
                    "correct": is_correct,
                    "failure_pattern": failure_pattern,
                    "meta": pos_df.loc[idx, "meta"],
                    "context": pos_df.loc[idx, "context"],
                    "source": source_list[idx],
                    "target": target_list[idx],
                }
            )
    pd.DataFrame(summary_rows).to_csv(test_dir / f"{dataset}_test_v_exclusion_summary.csv", index=False)
    slide_df = _preserve_text_columns(pd.DataFrame(slide_rows))
    slide_df.to_csv(test_dir / f"{dataset}_test_v_exclusion_examples.csv", index=False)
    for condition_name in slide_df["condition"].unique():
        condition_failures = slide_df[(slide_df["condition"] == condition_name) & (~slide_df["correct"])].copy()
        _write_error_analysis_csv(
            test_dir,
            f"{dataset}_test_v_{condition_name}_error_analysis.csv",
            condition_failures,
            "Test V",
            condition_name,
        )


def export_test_vi(
    output_dir: Path,
    dataset: str,
    dev_df: pd.DataFrame,
    generic_df: pd.DataFrame,
    codebook_list: list[dict],
) -> None:
    test_dir = _test_output_dir(output_dir, "test_vi")
    label_map = {item["Label"]: f"LABEL_{n+1}" for n, item in enumerate(codebook_list)}
    rows = dev_df.copy()
    rows["expected_generic_label"] = rows["label"].map(label_map).fillna("LABEL_NA")
    rows["model_predicted_label"] = generic_df["prediction"]
    rows["model_raw_output"] = generic_df["raw_prediction"]
    rows["correct"] = rows["expected_generic_label"] == rows["model_predicted_label"]
    rows = _preserve_text_columns(rows)
    rows.to_csv(test_dir / f"{dataset}_test_vi_generic_label_examples.csv", index=False)
    _write_error_analysis_csv(
        test_dir,
        f"{dataset}_test_vi_generic_label_error_analysis.csv",
        rows[~rows["correct"]].copy(),
        "Test VI",
        "generic_label",
    )


def export_test_vii(
    output_dir: Path,
    dataset: str,
    dev_df: pd.DataFrame,
    swapped_df: pd.DataFrame,
    codebook_list: list[dict],
) -> None:
    test_dir = _test_output_dir(output_dir, "test_vii")
    label_list = [item["Label"] for item in codebook_list]
    swapped_labels = derangement_shuffle(label_list, random_state=42)
    label_map = {orig: new for orig, new in zip(label_list, swapped_labels)}
    rows = dev_df.copy()
    rows["expected_swapped_label"] = rows["label"].map(label_map).fillna("LABEL_NA")
    rows["model_predicted_label"] = swapped_df["prediction"]
    rows["model_raw_output"] = swapped_df["raw_prediction"]
    rows["correct"] = rows["expected_swapped_label"] == rows["model_predicted_label"]
    rows = _preserve_text_columns(rows)
    rows.to_csv(test_dir / f"{dataset}_test_vii_swapped_label_examples.csv", index=False)
    _write_error_analysis_csv(
        test_dir,
        f"{dataset}_test_vii_swapped_label_error_analysis.csv",
        rows[~rows["correct"]].copy(),
        "Test VII",
        "swapped_label",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export reviewer-friendly example CSVs for the paper behavioral probes."
    )
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument(
        "--dataset",
        required=True,
        type=_dataset_slug_arg,
        help="Dataset slug (bundled bfrs/ccc/manifestos/plover or custom; requires splits + codebook via env).",
    )
    parser.add_argument("--quantization", default="4", type=str)
    parser.add_argument("--limit", default=25, type=int, help="Number of dev examples to use. <=0 means full dev split.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "behavioral_examples")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = args.dataset
    output_dir = args.output_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.model_name, quantization=args.quantization)
    tokenizer = load_tokenizer(args.model_name)
    print("Model ready.")

    codebook_list, _, _ = load_codebook(dataset)
    dev_df = _dev_df(dataset, args.limit)

    original_df = _load_prediction_cache(args.model_name, args.quantization, model, tokenizer, dataset, args.limit, "original")
    reverse_df = _load_prediction_cache(args.model_name, args.quantization, model, tokenizer, dataset, args.limit, "reverse")
    shuffle_df = _load_prediction_cache(args.model_name, args.quantization, model, tokenizer, dataset, args.limit, "shuffle")
    generic_df = _load_prediction_cache(args.model_name, args.quantization, model, tokenizer, dataset, args.limit, "generic_label")
    swapped_df = _load_prediction_cache(args.model_name, args.quantization, model, tokenizer, dataset, args.limit, "swapped_label")

    export_test_i(output_dir, dataset, original_df, codebook_list)
    export_test_ii(output_dir, dataset, model, tokenizer, codebook_list)
    export_test_iv(output_dir, dataset, original_df, reverse_df, shuffle_df)
    export_test_vi(output_dir, dataset, dev_df, generic_df, codebook_list)
    export_test_vii(output_dir, dataset, dev_df, swapped_df, codebook_list)

    print(f"Saved example CSVs to {output_dir}")


if __name__ == "__main__":
    main()
