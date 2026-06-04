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
class RagRunConfig:
    dataset: str
    split_file: Path
    codebook_file: Path
    model_name: str
    quantization: str = "4"
    limit: int = 0
    batch_size: int = 2
    max_new_tokens: int = 64
    embed_model: str = "all-MiniLM-L6-v2"
    top_k_codebook: int = 4
    top_k_rules: int = 2
    top_k_examples: int = 3
    strategy: str = "cb"
    output_dir: Path = Path("results/ragv1_pipeline")
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


def read_labelled_split(split_file: Path, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(split_file)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{split_file} must contain text and label columns; got {list(df.columns)}")
    if limit and limit > 0:
        df = df.sample(min(int(limit), len(df)), random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def _format_chunk(row: Dict[str, str]) -> str:
    label = str(row.get("Label", "")).strip()
    definition = str(row.get("Definition", "")).strip()
    clarification = str(row.get("Clarification", "")).strip()
    pieces = [label]
    if clarification:
        pieces.append(clarification)
    if definition:
        pieces.append(definition)
    return ": ".join(pieces)


def codebook_to_chunks(codebook_list: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    chunks = []
    for i, row in enumerate(codebook_list):
        chunks.append({
            "id": f"cb_{i + 1}",
            "rootcode": str(row.get("Label", "")).strip(),
            "type": "definition",
            "text": _format_chunk(row),
            "row": dict(row),
        })
    return chunks


def codebook_to_rules(instruction_dict: Dict[str, str]) -> List[Dict[str, str]]:
    rules = []
    instruction = instruction_dict.get("Instruction", "").strip()
    if instruction:
        rules.append({"id": "instruction", "type": "disambiguation", "text": instruction})
    output = instruction_dict.get("Output Reminder", "").strip()
    if output:
        rules.append({"id": "output_reminder", "type": "disambiguation", "text": output})
    return rules


def codebook_to_examples(codebook_list: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    examples = []
    for row in codebook_list:
        label = str(row.get("Label", "")).strip()
        for key, kind in (("Positive Example", "positive"), ("Negative Example", "negative")):
            if row.get(key):
                examples.append({
                    "rootcode": label,
                    "type": kind,
                    "text": str(row[key]).strip(),
                    "explanation": f"{kind} example from codebook entry {label}",
                })
    return examples


class RagV1Retriever:
    """RAG-v1-style retriever over codebook chunks, rules, and examples."""

    def __init__(
        self,
        codebook_list: Sequence[Dict[str, str]],
        instruction_dict: Dict[str, str],
        *,
        embed_model: str = "all-MiniLM-L6-v2",
        top_k_codebook: int = 4,
        top_k_rules: int = 2,
        top_k_examples: int = 3,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install RAG dependencies first: pip install sentence-transformers") from exc

        self.embedder = SentenceTransformer(embed_model)
        self.top_k_codebook = top_k_codebook
        self.top_k_rules = top_k_rules
        self.top_k_examples = top_k_examples
        self.codebook_chunks = codebook_to_chunks(codebook_list)
        self.rules = codebook_to_rules(instruction_dict)
        self.examples = codebook_to_examples(codebook_list)
        self._rebuild_embeddings()

    def with_codebook(self, codebook_list: Sequence[Dict[str, str]], instruction_dict: Optional[Dict[str, str]] = None) -> "RagV1Retriever":
        new = object.__new__(RagV1Retriever)
        new.embedder = self.embedder
        new.top_k_codebook = self.top_k_codebook
        new.top_k_rules = self.top_k_rules
        new.top_k_examples = self.top_k_examples
        new.codebook_chunks = codebook_to_chunks(codebook_list)
        new.rules = codebook_to_rules(instruction_dict or {})
        new.examples = codebook_to_examples(codebook_list)
        new._rebuild_embeddings()
        return new

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        return self.embedder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def _rebuild_embeddings(self) -> None:
        self.cb_texts = [c["text"] for c in self.codebook_chunks]
        self.rule_texts = [r["text"] for r in self.rules]
        self.example_texts = [e["text"] for e in self.examples]
        self.cb_embs = self._encode(self.cb_texts)
        self.rule_embs = self._encode(self.rule_texts)
        self.example_embs = self._encode(self.example_texts)

    def _topk(self, query: str, texts: Sequence[str], embs: np.ndarray, k: int) -> List[int]:
        if not texts or embs.shape[0] == 0 or k <= 0:
            return []
        q = self._encode([query])[0]
        scores = embs @ q
        order = np.argsort(-scores)
        return [int(i) for i in order[: min(k, len(order))]]

    def retrieve(self, sentence: str, strategy: str = "cb") -> Dict[str, List[Dict[str, Any]]]:
        cb_idx = self._topk(sentence, self.cb_texts, self.cb_embs, self.top_k_codebook)
        rule_idx = self._topk(sentence, self.rule_texts, self.rule_embs, self.top_k_rules)
        ex_idx = []
        if strategy in ("cb_ex", "noisy"):
            ex_idx = self._topk(sentence, self.example_texts, self.example_embs, self.top_k_examples)
        return {
            "codebook_chunks": [self.codebook_chunks[i] for i in cb_idx],
            "rules": [self.rules[i] for i in rule_idx],
            "examples": [self.examples[i] for i in ex_idx],
        }


def build_rag_prompt(
    sentence: str,
    retriever: RagV1Retriever,
    labels: Sequence[str],
    *,
    strategy: str = "cb",
) -> str:
    retrieved = retriever.retrieve(sentence, strategy=strategy)
    parts = [
        "You are a political event classifier using the PLOVER ontology. "
        "Classify the relation between source (<S></S>) and target (<T></T>)."
    ]
    parts.append("\nRELEVANT LABEL DEFINITIONS:")
    for i, chunk in enumerate(retrieved["codebook_chunks"], 1):
        parts.append(f"{i}. {chunk['text']}")
    if retrieved["rules"]:
        parts.append("\nDISAMBIGUATION RULES (apply these when choosing between similar labels):")
        for rule in retrieved["rules"]:
            parts.append(f"- {rule['text']}")
    if retrieved["examples"]:
        parts.append("\nSIMILAR LABELED EXAMPLES:")
        for ex in retrieved["examples"]:
            parts.append(f"  Sentence: {ex['text']}")
            parts.append(f"  Label: {ex['rootcode']} - {ex['explanation']}")
    parts.append(f"\nVALID LABELS: {', '.join(labels)}")
    parts.append(f"\nSentence: {sentence}")
    parts.append("\nOutput ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")
    return "\n".join(parts)


def parse_rag_answers(raw_answers: Sequence[str], allowed_labels: Sequence[str]) -> List[Optional[str]]:
    return parse_answers(raw_answers, list(allowed_labels))


def make_rag_prompts(
    documents: Sequence[str],
    retriever: RagV1Retriever,
    tokenizer: Any,
    labels: Sequence[str],
    *,
    strategy: str,
    system_message: str = "",
) -> List[str]:
    return [
        make_prompt(tokenizer, system_message, build_rag_prompt(str(doc), retriever, labels, strategy=strategy))
        for doc in documents
    ]


def predict_rag(
    documents: Sequence[str],
    retriever: RagV1Retriever,
    model: Any,
    tokenizer: Any,
    labels: Sequence[str],
    *,
    strategy: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int = 42,
) -> Tuple[List[Optional[str]], List[str]]:
    prompts = make_rag_prompts(documents, retriever, tokenizer, labels, strategy=strategy)
    raw = batch_call_llm(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return parse_rag_answers(raw, labels), raw


def pessimistic_unknowns(predictions: Sequence[Optional[str]], gold: Sequence[str], labels: Sequence[str]) -> List[str]:
    out = []
    label_list = list(labels)
    for pred, true in zip(predictions, gold):
        if pred in label_list:
            out.append(str(pred))
            continue
        out.append(next((label for label in label_list if label != true), label_list[0]))
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
    """Fleiss' kappa for three prediction sets over the same items."""
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
        matrix.append([sum(pred == label for pred in row) for label in label_list])
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return float("nan")
    n_items, _ = arr.shape
    n_raters = 3.0
    p_i = (np.sum(arr * arr, axis=1) - n_raters) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_i))
    p_j = np.sum(arr, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j * p_j))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else float("nan")
    return float((p_bar - p_e) / (1.0 - p_e))


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


def shuffled_codebook(codebook_list: Sequence[Dict[str, str]], seed: int = 42) -> List[Dict[str, str]]:
    rows = [dict(row) for row in codebook_list]
    random.Random(seed).shuffle(rows)
    return rows


def run_task_eval(
    df: pd.DataFrame,
    retriever: RagV1Retriever,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: RagRunConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    predictions, raw = predict_rag(
        df["text"].astype(str).tolist(),
        retriever,
        model,
        tokenizer,
        labels,
        strategy=config.strategy,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    scored = df.copy()
    scored["prediction"] = predictions
    scored["prediction_raw"] = raw
    metrics = score_predictions(scored["label"].astype(str).tolist(), predictions, labels)
    metrics["unknown_rate"] = float(np.mean([p is None for p in predictions]))
    return scored, metrics


def _result(probe: str, metric: str, value: float, config: RagRunConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "behavioral_probe": probe,
        "metric": metric,
        "value": value,
        "dataset": config.dataset,
        "model_type": config.model_name,
        "quantization": config.quantization,
        "limit": config.limit,
        "prompt_style": f"ragv1_{config.strategy}",
        "codebook_file": str(config.codebook_file),
    }
    if extra:
        row.update(extra)
    return row


def run_rag_behavior_probes(
    df: pd.DataFrame,
    codebook_list: Sequence[Dict[str, str]],
    instruction_dict: Dict[str, str],
    retriever: RagV1Retriever,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: RagRunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    base_pred, base_raw = predict_rag(df["text"].astype(str).tolist(), retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    base_metrics = score_predictions(df["label"].astype(str).tolist(), base_pred, labels)
    for metric, value in base_metrics.items():
        records.append(_result("ragv1_dev_baseline", metric, value, config))
    records.append(_result("ragv1_legal_predictions", "legal_rate", float(np.mean([p in labels for p in base_pred])), config))
    frames.append(pd.DataFrame({"probe": "baseline", "text": df["text"], "label": df["label"], "prediction": base_pred, "raw": base_raw}))

    for probe, cb_variant in (
        ("ragv1_order_reversed", list(codebook_list)[::-1]),
        ("ragv1_order_shuffled", shuffled_codebook(codebook_list)),
    ):
        variant_retriever = retriever.with_codebook(cb_variant, instruction_dict)
        pred, raw = predict_rag(df["text"].astype(str).tolist(), variant_retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        metrics = score_predictions(df["label"].astype(str).tolist(), pred, labels)
        for metric, value in metrics.items():
            records.append(_result(probe, metric, value, config))
        records.append(_result(probe, "percent_change_from_baseline", float(np.mean([a != b for a, b in zip(base_pred, pred)])), config))
        frames.append(pd.DataFrame({"probe": probe, "text": df["text"], "label": df["label"], "prediction": pred, "raw": raw}))

    reverse_frame = frames[-2] if len(frames) >= 3 else None
    shuffle_frame = frames[-1] if len(frames) >= 3 else None
    if reverse_frame is not None and shuffle_frame is not None:
        records.append(_result(
            "ragv1_order_fleiss_kappa",
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
    gen_retriever = retriever.with_codebook(gen_codebook, instruction_dict)
    gen_pred, gen_raw = predict_rag(df["text"].astype(str).tolist(), gen_retriever, model, tokenizer, gen_labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    for metric, value in score_predictions(gen_gold, gen_pred, gen_labels).items():
        records.append(_result("ragv1_generic_labels", metric, value, config))
    frames.append(pd.DataFrame({"probe": "generic_labels", "text": df["text"], "label": gen_gold, "prediction": gen_pred, "raw": gen_raw}))

    swap_codebook, swap_map = swapped_label_codebook(codebook_list)
    swap_labels = labels_from_codebook(swap_codebook)
    swap_gold = [swap_map.get(str(x).strip(), "LABEL_NA") for x in df["label"].tolist()]
    swap_retriever = retriever.with_codebook(swap_codebook, instruction_dict)
    swap_pred, swap_raw = predict_rag(df["text"].astype(str).tolist(), swap_retriever, model, tokenizer, swap_labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    for metric, value in score_predictions(swap_gold, swap_pred, swap_labels).items():
        records.append(_result("ragv1_swapped_labels", metric, value, config))
    frames.append(pd.DataFrame({"probe": "swapped_labels", "text": df["text"], "label": swap_gold, "prediction": swap_pred, "raw": swap_raw}))

    definition_docs = [row.get("Definition", "") for row in codebook_list]
    if any(definition_docs):
        def_pred, def_raw = predict_rag(definition_docs, retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        for metric, value in score_predictions(labels, def_pred, labels).items():
            records.append(_result("ragv1_definition_recovery", metric, value, config))
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
        pos_pred, pos_raw = predict_rag(pos_docs, retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        for metric, value in score_predictions(pos_gold, pos_pred, labels).items():
            records.append(_result("ragv1_positive_examples", metric, value, config))
        frames.append(pd.DataFrame({"probe": "positive_examples", "text": pos_docs, "label": pos_gold, "prediction": pos_pred, "raw": pos_raw}))
    if neg_docs:
        neg_pred, neg_raw = predict_rag(neg_docs, retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        records.append(_result("ragv1_negative_examples", "not_original_label_rate", float(np.mean([p != g for p, g in zip(neg_pred, neg_gold)])), config))
        frames.append(pd.DataFrame({"probe": "negative_examples", "text": neg_docs, "label": neg_gold, "prediction": neg_pred, "raw": neg_raw}))

    return pd.DataFrame(records), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_pipeline_outputs(
    task_predictions: pd.DataFrame,
    task_metrics: Dict[str, float],
    behavior_results: pd.DataFrame,
    behavior_predictions: pd.DataFrame,
    config: RagRunConfig,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_part = config.model_name.split("/")[-1]
    stem = f"{config.dataset}_{config.split_file.stem}_{config.codebook_file.stem}_{config.strategy}_{model_part}_limit{config.limit}"
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

