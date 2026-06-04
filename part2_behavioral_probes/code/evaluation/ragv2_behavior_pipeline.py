from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.codebook_utils import batch_call_llm, load_model, load_tokenizer, make_prompt, parse_answers
from evaluation.rag_behavior_pipeline import (
    PLOVER_ROOT2QUAD,
    PLOVER_ROOTCODES,
    RagRunConfig,
    codebook_to_chunks,
    codebook_to_examples,
    codebook_to_rules,
    fleiss_kappa_three,
    generic_label_codebook,
    labels_from_codebook,
    load_structured_codebook,
    read_labelled_split,
    save_pipeline_outputs,
    score_predictions,
    shuffled_codebook,
    swapped_label_codebook,
)


QUAD_NAMES = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}

QUAD_ROOTCODES = {
    1: ["AGREE", "CONSULT", "SUPPORT"],
    2: ["COOPERATE", "AID", "YIELD"],
    3: ["REQUEST", "ACCUSE", "REJECT", "THREATEN"],
    4: ["PROTEST", "SANCTION", "MOBILIZE", "COERCE", "ASSAULT"],
}

CONFUSABLE_NEIGHBORS = {
    "AGREE": ["SUPPORT", "CONSULT"],
    "CONSULT": ["AGREE", "SUPPORT"],
    "SUPPORT": ["AGREE", "COOPERATE"],
    "COOPERATE": ["SUPPORT", "AID"],
    "AID": ["COOPERATE", "YIELD"],
    "YIELD": ["AID", "AGREE"],
    "REQUEST": ["ACCUSE", "PROTEST"],
    "ACCUSE": ["REQUEST", "REJECT"],
    "REJECT": ["ACCUSE", "THREATEN"],
    "THREATEN": ["REJECT", "MOBILIZE"],
    "PROTEST": ["REQUEST", "MOBILIZE"],
    "SANCTION": ["COERCE", "REJECT"],
    "MOBILIZE": ["THREATEN", "PROTEST"],
    "COERCE": ["ASSAULT", "SANCTION"],
    "ASSAULT": ["COERCE", "MOBILIZE"],
}

QUAD_DESCRIPTIONS = [
    "verbal cooperation agreement meeting consultation support endorsement diplomatic promise",
    "material cooperation aid supply exchange concession release humanitarian military assistance",
    "verbal conflict accusation demand request threat rejection warning criticism refusal",
    "material conflict protest sanction mobilize arrest violence attack coerce military force",
]


@dataclass
class RagV2RunConfig(RagRunConfig):
    embed_model: str = "all-mpnet-base-v2"
    top_k_rules: int = 3
    top_k_examples: int = 4
    strategy: str = "hier"
    output_dir: Path = Path("results/ragv2_pipeline")


class RagV2Retriever:
    """RAG-v2-style retriever: flat or hierarchical with confusable neighbors."""

    def __init__(
        self,
        codebook_list: Sequence[Dict[str, str]],
        instruction_dict: Dict[str, str],
        *,
        embed_model: str = "all-mpnet-base-v2",
        top_k_codebook: int = 4,
        top_k_rules: int = 3,
        top_k_examples: int = 4,
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
        self.cb_by_root = {c["rootcode"]: c for c in self.codebook_chunks if c.get("rootcode")}
        self._rebuild_embeddings()

    def with_codebook(self, codebook_list: Sequence[Dict[str, str]], instruction_dict: Optional[Dict[str, str]] = None) -> "RagV2Retriever":
        new = object.__new__(RagV2Retriever)
        new.embedder = self.embedder
        new.top_k_codebook = self.top_k_codebook
        new.top_k_rules = self.top_k_rules
        new.top_k_examples = self.top_k_examples
        new.codebook_chunks = codebook_to_chunks(codebook_list)
        new.rules = codebook_to_rules(instruction_dict or {})
        new.examples = codebook_to_examples(codebook_list)
        new.cb_by_root = {c["rootcode"]: c for c in new.codebook_chunks if c.get("rootcode")}
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
        self.quad_embs = self._encode(QUAD_DESCRIPTIONS)

    def _topk(self, query: str, texts: Sequence[str], embs: np.ndarray, k: int) -> List[int]:
        if not texts or embs.shape[0] == 0 or k <= 0:
            return []
        q = self._encode([query])[0]
        scores = embs @ q
        return [int(i) for i in np.argsort(-scores)[: min(k, len(texts))]]

    def predict_quadcode(self, sentence: str) -> int:
        q = self._encode([sentence])[0]
        sims = self.quad_embs @ q
        return int(np.argmax(sims)) + 1

    def retrieve_flat(self, sentence: str, include_examples: bool = False) -> Dict[str, Any]:
        cb_idx = self._topk(sentence, self.cb_texts, self.cb_embs, self.top_k_codebook)
        rule_idx = self._topk(sentence, self.rule_texts, self.rule_embs, self.top_k_rules)
        chunks = [self.codebook_chunks[i] for i in cb_idx]
        roots = {c["rootcode"] for c in chunks}
        for root in list(roots):
            for neighbor in CONFUSABLE_NEIGHBORS.get(root, []):
                if neighbor not in roots and neighbor in self.cb_by_root:
                    chunks.append(self.cb_by_root[neighbor])
                    roots.add(neighbor)
        result = {
            "codebook_chunks": chunks,
            "rules": [self.rules[i] for i in rule_idx],
            "examples": [],
        }
        if include_examples:
            ex_idx = self._topk(sentence, self.example_texts, self.example_embs, self.top_k_examples)
            result["examples"] = [self.examples[i] for i in ex_idx]
        return result

    def retrieve_hierarchical(self, sentence: str, include_examples: bool = False) -> Dict[str, Any]:
        predicted_quad = self.predict_quadcode(sentence)
        target_roots = set(QUAD_ROOTCODES.get(predicted_quad, PLOVER_ROOTCODES))
        for root in list(target_roots):
            for neighbor in CONFUSABLE_NEIGHBORS.get(root, []):
                if PLOVER_ROOT2QUAD.get(neighbor) != predicted_quad:
                    target_roots.add(neighbor)
        chunks = [self.cb_by_root[r] for r in target_roots if r in self.cb_by_root]
        rule_idx = self._topk(sentence, self.rule_texts, self.rule_embs, self.top_k_rules)
        result = {
            "codebook_chunks": chunks,
            "rules": [self.rules[i] for i in rule_idx],
            "examples": [],
            "predicted_quad": predicted_quad,
        }
        if include_examples:
            ex_idx = self._topk(sentence, self.example_texts, self.example_embs, self.top_k_examples * 2)
            candidates = [self.examples[i] for i in ex_idx]
            same_quad = [e for e in candidates if PLOVER_ROOT2QUAD.get(e.get("rootcode")) == predicted_quad]
            other = [e for e in candidates if PLOVER_ROOT2QUAD.get(e.get("rootcode")) != predicted_quad]
            result["examples"] = (same_quad + other)[: self.top_k_examples]
        return result

    def retrieve(self, sentence: str, strategy: str = "hier") -> Dict[str, Any]:
        if strategy in ("hier", "hier_ex", "noisy"):
            return self.retrieve_hierarchical(sentence, include_examples=strategy in ("hier_ex", "noisy"))
        return self.retrieve_flat(sentence, include_examples=strategy == "cb_ex")


def build_ragv2_prompt(sentence: str, retriever: RagV2Retriever, labels: Sequence[str], *, strategy: str = "hier") -> str:
    retrieved = retriever.retrieve(sentence, strategy=strategy)
    parts = [
        "You are a political event classifier using the PLOVER ontology.",
        f"\nSENTENCE TO CLASSIFY:\n{sentence}",
    ]
    if "predicted_quad" in retrieved:
        quad = retrieved["predicted_quad"]
        parts.append(f"\nPRE-CLASSIFICATION: This event is likely {QUAD_NAMES.get(quad, '?')}.")
        parts.append(f"Primary candidates: {', '.join(QUAD_ROOTCODES.get(quad, PLOVER_ROOTCODES))}")

    parts.append("\nRELEVANT LABEL DEFINITIONS:")
    seen = set()
    for i, chunk in enumerate(retrieved["codebook_chunks"], 1):
        root = chunk.get("rootcode")
        if root in seen:
            continue
        parts.append(f"{i}. {chunk['text']}")
        seen.add(root)

    if retrieved["rules"]:
        parts.append("\nDISAMBIGUATION RULES:")
        for rule in retrieved["rules"]:
            parts.append(f"- {rule['text']}")

    if retrieved["examples"]:
        parts.append("\nSIMILAR LABELED EXAMPLES:")
        for ex in retrieved["examples"]:
            parts.append(f"  Input: {ex['text']}")
            parts.append(f"  Label: {ex['rootcode']} ({ex['explanation']})")

    parts.append(f"\nVALID LABELS: {', '.join(labels)}")
    parts.append("\nOutput ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")
    return "\n".join(parts)


def predict_ragv2(
    documents: Sequence[str],
    retriever: RagV2Retriever,
    model: Any,
    tokenizer: Any,
    labels: Sequence[str],
    *,
    strategy: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int = 42,
) -> Tuple[List[Optional[str]], List[str]]:
    prompts = [
        make_prompt(tokenizer, "", build_ragv2_prompt(str(doc), retriever, labels, strategy=strategy))
        for doc in documents
    ]
    raw = batch_call_llm(prompts, model, tokenizer, batch_size=batch_size, max_new_tokens=max_new_tokens, seed=seed)
    return parse_answers(raw, list(labels)), raw


def run_task_eval(
    df: pd.DataFrame,
    retriever: RagV2Retriever,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: RagV2RunConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred, raw = predict_ragv2(
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
    scored["prediction"] = pred
    scored["prediction_raw"] = raw
    metrics = score_predictions(scored["label"].astype(str).tolist(), pred, labels)
    metrics["unknown_rate"] = float(np.mean([p is None for p in pred]))
    return scored, metrics


def _result(probe: str, metric: str, value: float, config: RagV2RunConfig) -> Dict[str, Any]:
    return {
        "behavioral_probe": probe,
        "metric": metric,
        "value": value,
        "dataset": config.dataset,
        "model_type": config.model_name,
        "quantization": config.quantization,
        "limit": config.limit,
        "prompt_style": f"ragv2_{config.strategy}",
        "codebook_file": str(config.codebook_file),
    }


def run_ragv2_behavior_probes(
    df: pd.DataFrame,
    codebook_list: Sequence[Dict[str, str]],
    instruction_dict: Dict[str, str],
    retriever: RagV2Retriever,
    labels: Sequence[str],
    model: Any,
    tokenizer: Any,
    config: RagV2RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    base_pred, base_raw = predict_ragv2(df["text"].astype(str).tolist(), retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    for metric, value in score_predictions(df["label"].astype(str).tolist(), base_pred, labels).items():
        records.append(_result("ragv2_dev_baseline", metric, value, config))
    records.append(_result("ragv2_legal_predictions", "legal_rate", float(np.mean([p in labels for p in base_pred])), config))
    frames.append(pd.DataFrame({"probe": "baseline", "text": df["text"], "label": df["label"], "prediction": base_pred, "raw": base_raw}))

    variant_preds = {}
    for probe, cb_variant in (
        ("ragv2_order_reversed", list(codebook_list)[::-1]),
        ("ragv2_order_shuffled", shuffled_codebook(codebook_list)),
    ):
        variant_retriever = retriever.with_codebook(cb_variant, instruction_dict)
        pred, raw = predict_ragv2(df["text"].astype(str).tolist(), variant_retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        variant_preds[probe] = pred
        for metric, value in score_predictions(df["label"].astype(str).tolist(), pred, labels).items():
            records.append(_result(probe, metric, value, config))
        records.append(_result(probe, "percent_change_from_baseline", float(np.mean([a != b for a, b in zip(base_pred, pred)])), config))
        frames.append(pd.DataFrame({"probe": probe, "text": df["text"], "label": df["label"], "prediction": pred, "raw": raw}))

    if "ragv2_order_reversed" in variant_preds and "ragv2_order_shuffled" in variant_preds:
        records.append(_result("ragv2_order_fleiss_kappa", "fleiss_kappa", fleiss_kappa_three(base_pred, variant_preds["ragv2_order_reversed"], variant_preds["ragv2_order_shuffled"], labels), config))

    gen_codebook, gen_map = generic_label_codebook(codebook_list)
    gen_labels = labels_from_codebook(gen_codebook)
    gen_gold = [gen_map.get(str(x).strip(), "LABEL_NA") for x in df["label"].tolist()]
    gen_retriever = retriever.with_codebook(gen_codebook, instruction_dict)
    gen_pred, gen_raw = predict_ragv2(df["text"].astype(str).tolist(), gen_retriever, model, tokenizer, gen_labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    for metric, value in score_predictions(gen_gold, gen_pred, gen_labels).items():
        records.append(_result("ragv2_generic_labels", metric, value, config))
    frames.append(pd.DataFrame({"probe": "generic_labels", "text": df["text"], "label": gen_gold, "prediction": gen_pred, "raw": gen_raw}))

    swap_codebook, swap_map = swapped_label_codebook(codebook_list)
    swap_labels = labels_from_codebook(swap_codebook)
    swap_gold = [swap_map.get(str(x).strip(), "LABEL_NA") for x in df["label"].tolist()]
    swap_retriever = retriever.with_codebook(swap_codebook, instruction_dict)
    swap_pred, swap_raw = predict_ragv2(df["text"].astype(str).tolist(), swap_retriever, model, tokenizer, swap_labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
    for metric, value in score_predictions(swap_gold, swap_pred, swap_labels).items():
        records.append(_result("ragv2_swapped_labels", metric, value, config))
    frames.append(pd.DataFrame({"probe": "swapped_labels", "text": df["text"], "label": swap_gold, "prediction": swap_pred, "raw": swap_raw}))

    definition_docs = [row.get("Definition", "") for row in codebook_list]
    if any(definition_docs):
        def_pred, def_raw = predict_ragv2(definition_docs, retriever, model, tokenizer, labels, strategy=config.strategy, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)
        for metric, value in score_predictions(labels, def_pred, labels).items():
            records.append(_result("ragv2_definition_recovery", metric, value, config))
        frames.append(pd.DataFrame({"probe": "definition_recovery", "text": definition_docs, "label": labels, "prediction": def_pred, "raw": def_raw}))

    return pd.DataFrame(records), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

