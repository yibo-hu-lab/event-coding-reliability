from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.paths import CODEBOOK_DIR, DATASET_SPLITS_DIR, RAW_DATA_DIR

_DATASET_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
STANDARD_COLS = ["text", "label", "meta", "context", "source", "target"]
RAW_EXPORT_FILENAMES = {
    "bfrs": "bfrs_reconstructed_from_splits.csv",
    "ccc": "ccc_reconstructed_from_splits.csv",
    "manifestos": "manifestos_reconstructed_from_splits.csv",
    "plover": "plover.csv",
}
MANIFESTOS_FILES = (
    "62623_201510.csv",
    "181980_201905.csv",
    "181210_201405.csv",
    "181420_200904.csv",
    "53951_201602.csv",
    "63320_201008.csv",
    "62420_201510.csv",
    "64951_198407.csv",
    "51901_201505.csv",
    "51421_201505.csv",
    "51421_199705.csv",
    "53110_201602.csv",
    "53110_201102.csv",
)


def _default_new_codebook_path(dataset: str) -> Path:
    dataset = dataset.lower()
    if dataset == "ccc":
        return CODEBOOK_DIR / "ccc_codebook_new_format.txt"
    if dataset == "bfrs":
        return CODEBOOK_DIR / "bfrs_codebook_new_format.txt"
    if dataset == "manifestos":
        return CODEBOOK_DIR / "manifesto_codebook_new_hand.txt"
    if dataset == "plover":
        return CODEBOOK_DIR / "plover_enriched_codebook.txt"
    if dataset == "aw":
        return CODEBOOK_DIR / "aw_enriched_codebook.txt"
    raise ValueError(f"Unsupported dataset for codebook parsing: {dataset}")


def _parse_new_codebook_format_light(dataset: str) -> tuple[list[dict], dict]:
    codebook_file = _default_new_codebook_path(dataset)
    with open(codebook_file, "r", encoding="utf-8") as f:
        codebook = f.read()

    instructions, codebook = codebook.split("### Categories ###")
    instructions = instructions.strip().replace("### Instructions ###", "").strip().split("\n")
    instruction_dict = {}
    for line in instructions:
        key, value = line.split(":", 1)
        if value.strip():
            instruction_dict[key.strip()] = value.strip()

    codebook_list = []
    for section in codebook.split("\n\n"):
        try:
            section_list = section.strip().split("\n")
            section_dict = {"Category": section_list[0].strip()}
            for line in section_list[1:]:
                line = line.strip()
                if line.startswith("--") or not line:
                    continue
                key, value = line.split(":", 1)
                if value.strip():
                    section_dict[key.strip()] = value.strip()
            codebook_list.append(section_dict)
        except ValueError:
            continue
    codebook_list = [item for item in codebook_list if "Label" in item]
    return codebook_list, instruction_dict


def _parse_manifestos_light(fn: str, party_info_df: pd.DataFrame) -> pd.DataFrame:
    fn_last = os.path.basename(fn)
    party_id, _ = fn_last.split("_")
    party_info = party_info_df[party_info_df["party"] == int(party_id)].to_dict("records")[0]
    df = pd.read_csv(fn)
    if "text" not in df.columns:
        df["text"] = df["content"]
        del df["content"]
    df = df.reset_index(drop=True)
    df["meta"] = f"an excerpt from a political party in {party_info['countryname']}."
    df["context"] = ""
    for n, _row in df.iterrows():
        if n == 0:
            df.at[n, "context"] = ""
        elif n == 1:
            df.at[n, "context"] = df.at[n - 1, "text"]
        else:
            df.at[n, "context"] = df.at[n - 2, "text"] + " " + df.at[n - 1, "text"]
    return df


def _require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}\n{hint}")
    return path


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_actor_text(value) -> str:
    text = _clean_text(value)
    text = text.strip(" ,;:-")
    if not text:
        return ""
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    return text.strip()


def _first_capitalized_spans(text: str) -> list[str]:
    matches = re.findall(r"\b(?:[A-Z][\w'./-]*)(?:\s+(?:[A-Z][\w'./-]*|of|and|the|for|to|in|on|at|de|la|le))*", text)
    seen = []
    for match in matches:
        cleaned = _normalize_actor_text(match)
        if cleaned and cleaned not in seen and len(cleaned) > 1:
            seen.append(cleaned)
    return seen


def _extract_source_target_from_text(text: str) -> tuple[str, str]:
    entities = _first_capitalized_spans(text)
    if len(entities) >= 2:
        return entities[0], entities[1]
    if len(entities) == 1:
        return entities[0], ""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'/-]+", text) if len(w) > 2]
    if len(words) >= 2:
        return words[0], words[1]
    if len(words) == 1:
        return words[0], ""
    return "", ""


def _populate_source_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = ""
    if "target" not in out.columns:
        out["target"] = ""
    for idx, row in out.iterrows():
        source = _normalize_actor_text(row.get("source", ""))
        target = _normalize_actor_text(row.get("target", ""))
        if not source or not target:
            inferred_source, inferred_target = _extract_source_target_from_text(_clean_text(row.get("text", "")))
            if not source:
                source = inferred_source
            if not target:
                target = inferred_target if inferred_target != source else ""
        out.at[idx, "source"] = source
        out.at[idx, "target"] = target
    return out


def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STANDARD_COLS:
        if col not in out.columns:
            out[col] = ""
    for col in STANDARD_COLS:
        out[col] = out[col].fillna("").astype(str)
    out["text"] = out["text"].map(_clean_text)
    out["label"] = out["label"].map(_clean_text)
    out["meta"] = out["meta"].map(_clean_text)
    out["context"] = out["context"].map(_clean_text)
    out["source"] = out["source"].map(_normalize_actor_text)
    out["target"] = out["target"].map(_normalize_actor_text)
    out = out[(out["text"] != "") & (out["label"] != "")].copy()
    out = _populate_source_target(out)
    return out[STANDARD_COLS]


def _raw_export_filename(dataset: str) -> str:
    """Unified CSV basename under raw_data (bundled names + arbitrary slugs like cameo)."""
    dataset = dataset.strip().lower()
    return RAW_EXPORT_FILENAMES.get(dataset, f"{dataset}.csv")


def _export_raw_like_file(dataset: str, df: pd.DataFrame, raw_data_dir: Path) -> Path:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    export_path = raw_data_dir / _raw_export_filename(dataset)
    _finalize_columns(df).to_csv(export_path, index=False)
    return export_path


def _load_reconstructed_raw(dataset: str, raw_data_dir: Path) -> pd.DataFrame:
    path = raw_data_dir / _raw_export_filename(dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    return _finalize_columns(pd.read_csv(path))


def _default_raw_csv_path(dataset: str, raw_data_dir: Path) -> Path:
    """Canonical RAW CSV path for a slug (no glob or filename priority)."""
    dataset = dataset.strip().lower()
    return raw_data_dir / _raw_export_filename(dataset)


def _load_dataset_from_named_csv(dataset: str, raw_data_dir: Path) -> pd.DataFrame:
    path = _default_raw_csv_path(dataset, raw_data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing raw CSV for {dataset!r}: {path}. "
            f"Add that file or set BEHAVIOR_{dataset.upper()}_RAW_DATA_FILE (legacy DATAVERSE_*) "
            "or pass --raw-data-file when building splits."
        )
    if dataset == "plover":
        return _load_plover_from_csv(path)
    return _finalize_columns(pd.read_csv(path))


def _env_raw_data_override(dataset: str) -> Path | None:
    for prefix in ("BEHAVIOR", "DATAVERSE"):
        env_value = os.environ.get(f"{prefix}_{dataset.upper()}_RAW_DATA_FILE", "").strip()
        if env_value:
            return Path(os.path.expanduser(env_value)).resolve()
    return None


def _reconstruct_from_existing_splits(dataset: str, splits_dir: Path = DATASET_SPLITS_DIR) -> pd.DataFrame:
    parts = []
    for split in ("train", "dev", "test"):
        path = splits_dir / f"{dataset}_{split}.csv"
        if path.is_file():
            parts.append(pd.read_csv(path))
    eval_path = splits_dir / f"{dataset}_eval.csv"
    if eval_path.is_file() and not (splits_dir / f"{dataset}_test.csv").is_file():
        parts.append(pd.read_csv(eval_path))
    if not parts:
        raise FileNotFoundError(f"No existing split files found for {dataset} under {splits_dir}")
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["text", "label", "meta", "context"], keep="first")
    return _finalize_columns(merged)


def _load_jsonl_records(path: Path) -> list[dict]:
    import jsonlines

    with jsonlines.open(path, "r") as reader:
        return list(reader.iter())


def _load_bfrs(raw_data_dir: Path) -> pd.DataFrame:
    codebook_list, _ = _parse_new_codebook_format_light("bfrs")
    src = _require_file(
        raw_data_dir / "BFRS" / "PK_Political_Violence_Codesheet_V10 (03JUN2013).xls",
        "Expected the original BFRS spreadsheet under `data/raw_data/BFRS/`.",
    )
    df = pd.read_excel(src)
    label_convert = {item["Category"]: item["Label"] for item in codebook_list if "Label" in item}
    df["text"] = df["Description"]
    df["label"] = df["Event"].map(label_convert)
    df["meta"] = "a news story from Pakistan."
    df["context"] = ""
    return _finalize_columns(df)


def _load_ccc(raw_data_dir: Path) -> pd.DataFrame:
    src = _require_file(
        raw_data_dir / "CCC" / "ccc_text_merged.jsonl",
        "Expected `data/raw_data/CCC/ccc_text_merged.jsonl`.",
    )
    data = _load_jsonl_records(src)
    codebook_list, _ = _parse_new_codebook_format_light("ccc")
    legal_labels = {str(item["Label"]).strip().upper() for item in codebook_list if "Label" in item}
    clean_rows = []
    for row in data:
        raw_label = row.get("type")
        text = _clean_text(row.get("text", ""))
        if not isinstance(raw_label, str) or not text:
            continue
        label_list = [item.strip() for item in raw_label.upper().split(",") if item.strip()]
        if not label_list or len(label_list) > 1:
            continue
        if len(set(label_list).intersection(legal_labels)) != len(set(label_list)):
            continue
        if len(text.split()) < 100 or len(text.split()) > 1000:
            continue
        if re.search(r"log in|register|subscript|subscrib", text[:150].lower()):
            continue
        clean_rows.append(
            {
                "text": text,
                "label": label_list[0],
                "meta": "a news story from the United States.",
                "context": "",
                "source": row.get("source", ""),
                "target": row.get("target", ""),
            }
        )
    return _finalize_columns(pd.DataFrame(clean_rows))


def _load_manifestos(raw_data_dir: Path) -> pd.DataFrame:
    codebook_list, _ = _parse_new_codebook_format_light("manifestos")
    codebook_src = _require_file(
        raw_data_dir / "Manifestos" / "codebook_categories_MPDS2020a.csv",
        "Expected the Manifestos raw bundle under `data/raw_data/Manifestos/`.",
    )
    party_info_src = _require_file(
        raw_data_dir / "Manifestos" / "parties_long_MPDataset_MPDS2024a.csv",
        "Expected `parties_long_MPDataset_MPDS2024a.csv` under `data/raw_data/Manifestos/`.",
    )
    structured_codebook = pd.read_csv(codebook_src)
    party_info_df = pd.read_csv(party_info_src)
    label_convert = {item["code"]: item["label"] for item in structured_codebook.to_dict("records")}
    label_category = {item["code"]: item["domain_name"] for item in structured_codebook.to_dict("records")}

    df_list = []
    for filename in MANIFESTOS_FILES:
        source = _require_file(
            raw_data_dir / "Manifestos" / filename,
            "One or more manifesto input CSV files are missing from `data/raw_data/Manifestos/`.",
        )
        df_list.append(_parse_manifestos_light(str(source), party_info_df))
    df = pd.concat(df_list, ignore_index=True)
    df["label"] = df["cmp_code"].map(label_convert)
    codebook_list = [item for item in codebook_list if item["Category"] in df["label"].unique()]
    label_conversion = {item["Category"]: item["Label"] for item in codebook_list if "Label" in item}
    df["label_category"] = df["cmp_code"].map(label_category)
    df["label"] = df["label"].map(label_conversion).fillna("NA")
    if "meta" not in df.columns:
        df["meta"] = ""
    if "context" not in df.columns:
        df["context"] = ""
    if "source" not in df.columns:
        df["source"] = df["meta"]
    if "target" not in df.columns:
        df["target"] = ""
    for col in ("cmp_code", "eu_code", "label_category"):
        if col in df.columns:
            del df[col]
    return _finalize_columns(df)


def _extract_actor_strings(value) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                actor_text = item.get("actorText", "")
                if actor_text:
                    out.append(_normalize_actor_text(actor_text))
            elif isinstance(item, str):
                cleaned = _normalize_actor_text(item)
                if cleaned:
                    out.append(cleaned)
        return [x for x in out if x]
    if isinstance(value, dict):
        actor_text = value.get("actorText", "")
        return [_normalize_actor_text(actor_text)] if actor_text else []
    cleaned = _normalize_actor_text(value)
    return [cleaned] if cleaned else []


def _load_plover_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    if "source_clean" in out.columns:
        out["source"] = out["source_clean"].fillna(out.get("source", ""))
    if "target_clean" in out.columns:
        out["target"] = out["target_clean"].fillna(out.get("target", ""))
    if "use_for_relation_classification" in out.columns:
        keep_mask = out["use_for_relation_classification"].fillna("").astype(str).str.strip().str.upper() == "YES"
        out = out[keep_mask].copy()
    for col in ("meta", "context"):
        if col not in out.columns:
            out[col] = ""
    out = _finalize_columns(out)
    out["text"] = out["text"].str.replace("{", "", regex=False).str.replace("}", "", regex=False).str.strip()
    single_pair_mask = (
        (out["source"] != "")
        & (out["target"] != "")
        & ~out["source"].str.contains(";", regex=False)
        & ~out["target"].str.contains(";", regex=False)
        & ~out["source"].str.contains("...", regex=False)
        & ~out["target"].str.contains("...", regex=False)
    )
    return out[single_pair_mask].reset_index(drop=True)


def _load_plover(raw_data_dir: Path) -> pd.DataFrame:
    codebook_list, _ = _parse_new_codebook_format_light("plover")
    legal_labels = {str(item["Label"]).strip().upper() for item in codebook_list if "Label" in item}
    raw_path = _require_file(
        raw_data_dir / "PLOVER_GSR_CAMEO.txt",
        "Expected `data/raw_data/PLOVER_GSR_CAMEO.txt` or a normalized CSV such as `data/raw_data/plover.csv`.",
    )
    with raw_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for row in records:
        if not isinstance(row, dict):
            continue
        label = _clean_text(row.get("event", ""))
        text = _clean_text(row.get("text", row.get("eventText", "")))
        if not text or label in ("", "DOCUMENT") or label not in legal_labels:
            continue
        source_values = _extract_actor_strings(row.get("source", ""))
        target_values = _extract_actor_strings(row.get("target", ""))
        source = source_values[0] if source_values else ""
        if target_values:
            target = target_values[0]
        elif len(source_values) > 1:
            target = source_values[1]
        else:
            target = ""
        rows.append(
            {
                "text": text,
                "label": label,
                "meta": "PLOVER event record.",
                "context": _clean_text(row.get("context", "")),
                "source": source,
                "target": target,
            }
        )
    return _finalize_columns(pd.DataFrame(rows))


DATASET_LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "bfrs": _load_bfrs,
    "ccc": _load_ccc,
    "manifestos": _load_manifestos,
    "plover": _load_plover,
}


def load_dataset_frame(dataset: str, raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    dataset = dataset.strip().lower()
    # Env / CLI (--raw-data-file) overrides must work for arbitrary slugs (not only ALLOWED presets).
    raw_override = _env_raw_data_override(dataset)
    if raw_override is not None:
        if not raw_override.is_file():
            raise FileNotFoundError(
                f"Raw-data override for {dataset!r} does not exist: {raw_override}"
            )
        if dataset == "plover":
            return _load_plover_from_csv(raw_override)
        return _finalize_columns(pd.read_csv(raw_override))
    if dataset not in DATASET_LOADERS:
        try:
            return _load_dataset_from_named_csv(dataset, raw_data_dir)
        except FileNotFoundError as exc:
            expected = _default_raw_csv_path(dataset, raw_data_dir)
            raise ValueError(
                f"Unsupported dataset {dataset!r}: expected raw file at {expected} or set "
                f"BEHAVIOR_{dataset.upper()}_RAW_DATA_FILE (legacy DATAVERSE_*), "
                "or pass --raw-data-file with exactly one slug."
            ) from exc

    try:
        df = _load_dataset_from_named_csv(dataset, raw_data_dir)
    except FileNotFoundError:
        loader = DATASET_LOADERS[dataset]
        try:
            df = loader(raw_data_dir)
        except FileNotFoundError:
            try:
                df = _load_reconstructed_raw(dataset, raw_data_dir)
            except FileNotFoundError:
                df = _reconstruct_from_existing_splits(dataset)
                _export_raw_like_file(dataset, df, raw_data_dir)
    return _finalize_columns(df)


def _stratify_labels_or_none(df: pd.DataFrame):
    counts = df["label"].value_counts()
    return df["label"] if not counts.empty and counts.min() >= 2 else None


def split_dataset(
    df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = _stratify_labels_or_none(df)
    train_df, heldout_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=stratify,
    )
    heldout_stratify = _stratify_labels_or_none(heldout_df)
    dev_df, test_df = train_test_split(
        heldout_df,
        test_size=0.50,
        random_state=seed,
        stratify=heldout_stratify,
    )
    return (
        train_df[STANDARD_COLS].reset_index(drop=True),
        dev_df[STANDARD_COLS].reset_index(drop=True),
        test_df[STANDARD_COLS].reset_index(drop=True),
    )


def write_splits(
    dataset: str,
    output_dir: Path = DATASET_SPLITS_DIR,
    raw_data_dir: Path = RAW_DATA_DIR,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    dataset = dataset.strip().lower()
    df = load_dataset_frame(dataset, raw_data_dir=raw_data_dir)
    _export_raw_like_file(dataset, df, raw_data_dir)
    train_df, dev_df, test_df = split_dataset(df, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{dataset}_train.csv"
    dev_path = output_dir / f"{dataset}_dev.csv"
    test_path = output_dir / f"{dataset}_test.csv"
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, dev_path, test_path


def _parse_datasets(value: str) -> Iterable[str]:
    """Comma-separated lowercase slugs; must be explicitly listed (no magic `all`)."""
    value = value.strip().lower()
    if not value:
        return []
    if value == "all":
        raise ValueError(
            "Magic token 'all' is no longer supported. Pass an explicit comma-separated "
            "list of slug(s) (e.g. cameo or bfrs,ccc)."
        )
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _validate_dataset_slug(name: str, parser: argparse.ArgumentParser) -> None:
    if len(name) >= 64 or not _DATASET_SLUG_RE.fullmatch(name):
        parser.error(
            f"Invalid dataset slug {name!r}. Use lowercase letters/digits plus optional '-' or '_' (max 63 chars)."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build unified train/dev/test CSV splits with text/label/meta/context/source/target columns."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated slug(s); required. CSV under --raw-data-dir, or BEHAVIOR_<SLUG>_RAW_DATA_FILE "
        "(legacy DATAVERSE_* supported), or built-in loaders for older presets.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw dataset folders or reconstructed raw-like CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_SPLITS_DIR,
        help="Directory to write *_train.csv, *_dev.csv, *_test.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raw = (args.datasets or "").strip()
    if not raw:
        parser.error("--datasets is required (comma-separated slug(s); there is no default batch).")
    try:
        datasets = list(_parse_datasets(raw))
    except ValueError as exc:
        parser.error(str(exc))
    for dataset in datasets:
        _validate_dataset_slug(dataset, parser)
    for dataset in datasets:
        train_path, dev_path, test_path = write_splits(
            dataset,
            output_dir=args.output_dir,
            raw_data_dir=args.raw_data_dir,
            seed=args.seed,
        )
        print(
            f"[make_dataset_splits] {dataset}: wrote\n"
            f"  {train_path}\n  {dev_path}\n  {test_path}\n"
            f"  raw-like: {args.raw_data_dir / _raw_export_filename(dataset)}"
        )


if __name__ == "__main__":
    main()
