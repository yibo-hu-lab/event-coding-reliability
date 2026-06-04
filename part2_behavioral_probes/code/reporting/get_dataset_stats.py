from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.codebook_utils import load_codebook
from core.paths import DATASET_SPLITS_DIR, DATASET_STATS_DIR

DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"


def discover_datasets_with_complete_splits(split_dir: Path = DATASET_SPLITS_DIR) -> list[str]:
    """Slug names that have train, dev, and test CSVs under ``split_dir``."""
    stems = [p.stem for p in split_dir.glob("*_*.csv")]
    trains = {s[: -len("_train")] for s in stems if s.endswith("_train")}
    devs = {s[: -len("_dev")] for s in stems if s.endswith("_dev")}
    tests = {s[: -len("_test")] for s in stems if s.endswith("_test")}
    return sorted(trains & devs & tests)


def _read_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATASET_SPLITS_DIR / f"{dataset}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def _load_tokenizer(tokenizer_name: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        print(
            f"[get_dataset_stats] Could not load tokenizer {tokenizer_name!r}; "
            f"falling back to whitespace counts only. ({exc})"
        )
        return None


def _codebook_lengths(dataset: str, tokenizer) -> tuple[float, float, int | None]:
    codebook_list, _, _ = load_codebook(dataset)
    definition_lengths = []
    total_lengths = []
    all_text = []
    for item in codebook_list:
        definition = str(item.get("Definition", "")).strip()
        base_len = len(definition.split())
        definition_lengths.append(base_len)
        section_len = base_len
        if definition:
            all_text.append(definition)
        for key in (
            "Clarification",
            "Negative Clarification",
            "Positive Example",
            "Negative Example",
        ):
            value = str(item.get(key, "")).strip()
            if not value:
                continue
            section_len += len(value.split())
            all_text.append(value)
        total_lengths.append(section_len)
    token_length = None
    if tokenizer is not None:
        tokenized = tokenizer(" ".join(all_text), return_tensors="pt", padding=False, truncation=False)
        token_length = int(len(tokenized["input_ids"][0]))
    return float(np.median(definition_lengths)), float(np.sum(total_lengths)), token_length


def get_dataset_stats(dataset: str, tokenizer) -> dict:
    train_df = _read_split(dataset, "train")
    dev_df = _read_split(dataset, "dev")
    test_df = _read_split(dataset, "test")
    med_desc, total_desc, llama_tokens = _codebook_lengths(dataset, tokenizer)
    doc_lens = [len(str(text).split()) for text in train_df["text"].fillna("")]
    return {
        "dataset": dataset,
        "num_categories": int(train_df["label"].nunique()),
        "median_definition_length": med_desc,
        "total_codebook_length": total_desc,
        "tokenizer_length": llama_tokens if llama_tokens is not None else -1,
        "median_document_length": float(np.median(doc_lens)) if doc_lens else 0.0,
        "num_train": int(len(train_df)),
        "num_dev": int(len(dev_df)),
        "num_test": int(len(test_df)),
    }


def write_dataset_descriptions(datasets: Iterable[str], tokenizer_name: str) -> Path:
    tokenizer = _load_tokenizer(tokenizer_name)
    rows = [get_dataset_stats(dataset, tokenizer) for dataset in datasets]
    stats_df = pd.DataFrame(rows).set_index("dataset").T
    DATASET_STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATASET_STATS_DIR / "dataset_descriptive_stats.csv"
    stats_df.to_csv(out_csv, index=True)
    return out_csv


def majority_class_baseline(dataset: str) -> Path:
    train_df = _read_split(dataset, "train")
    test_df = _read_split(dataset, "test")
    majority_label = train_df["label"].value_counts().idxmax()
    test_df = test_df[~test_df["label"].isna()].copy()
    test_df["prediction"] = majority_label
    report = classification_report(
        test_df["label"],
        test_df["prediction"],
        zero_division=0,
    )
    DATASET_STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASET_STATS_DIR / f"{dataset}_majority_class_baseline.txt"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"Majority class baseline for {dataset}\n")
        handle.write(f"Majority class: {majority_label}\n\n")
        handle.write(report)
    return out_path


def generate_latex_table(stats_csv: Path) -> Path:
    stats_df = pd.read_csv(stats_csv, index_col=0)
    cols = sorted(c for c in stats_df.columns if str(c).strip())
    stats_df = stats_df[cols]
    formatted_df = stats_df.copy()
    if "tokenizer_length" in formatted_df.index:
        formatted_df.loc["tokenizer_length"] = formatted_df.loc["tokenizer_length"].replace(-1, "N/A")
    latex = formatted_df.to_latex()
    DATASET_STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASET_STATS_DIR / "table_1_dataset_descriptives.tex"
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset descriptives and majority baselines.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="auto",
        help="Comma-separated dataset slugs, or the single keyword 'auto' to use every slug with "
        "complete train/dev/test CSVs under data/dataset_splits/.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer used to estimate codebook token lengths.",
    )
    parser.add_argument(
        "--write-latex",
        action="store_true",
        help="Also write a LaTeX table under results/dataset_stats/.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    spec = args.datasets.strip()
    pieces = [p.strip() for p in spec.split(",") if p.strip()]
    if len(pieces) == 1 and pieces[0].lower() == "auto":
        datasets = discover_datasets_with_complete_splits()
        if not datasets:
            print("[get_dataset_stats] No slug has train+dev+test CSVs under dataset_splits.", file=sys.stderr)
            sys.exit(1)
        print(f"[get_dataset_stats] auto-selected slugs ({len(datasets)}): {', '.join(datasets)}")
    else:
        datasets = pieces
        if not datasets:
            parser.error("--datasets must list at least one slug, or pass 'auto'.")
    stats_csv = write_dataset_descriptions(datasets, tokenizer_name=args.tokenizer_name)
    print(f"[get_dataset_stats] wrote {stats_csv}")
    for dataset in datasets:
        baseline_path = majority_class_baseline(dataset)
        print(f"[get_dataset_stats] wrote {baseline_path}")
    if args.write_latex:
        latex_path = generate_latex_table(stats_csv)
        print(f"[get_dataset_stats] wrote {latex_path}")


if __name__ == "__main__":
    main()
