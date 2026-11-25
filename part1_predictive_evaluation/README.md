# Part I: Predictive Evaluation

This directory contains the code for the predictive experiments in the paper. These experiments ask whether more detailed codebook guidance improves LLM event coding under the original codebook. Results are reported as root-level macro-F1 on PLV (15 PLOVER root labels, source-target event coding) and binary macro-F1 on AW (Cooperation vs. Conflict). The rule-adherence probes are in [`../part2_behavioral_probes/`](../part2_behavioral_probes/).

## Prompting Conditions

All conditions share the same label space within each dataset and differ only in the guidance given to the model.

| Condition | Guidance in the prompt | `--step` |
|---|---|---|
| No Codebook | Label names only | `llm_no_cb` |
| Compact | Label names, short definitions, and quad-level groupings | `llm_compact` |
| Enriched | Compact plus examples, event-mode guidance, boundary notes, and disambiguation rules | `llm_enriched` |
| ICL | Labeled demonstrations without full codebook definitions | `llm_icl` |
| CoT | Codebook prompt with a step-by-step reasoning instruction before the final label | `llm_cot` (PLV), `llm_cot_cb` (AW) |
| RAG | Retrieved definitions, examples, and rules from the Enriched codebook | `rag_cb` (in `supplemental/rag/`) |

## Setup

```bash
pip install transformers torch pandas scikit-learn requests
pip install sentence-transformers faiss-cpu   # RAG only
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
```

The paper reports means across these four open-source models. All scripts query a local Ollama server. The default model is `gemma2:9b`. To use another model, change `LLM_MODEL` at the top of the PLV scripts, or pass `--model` to the AW and RAG scripts.

The main predictive runs and the direct binary/quad runs use `temperature = 0.0`. The RAG scripts use `temperature = 0.1`. Each condition is run once per model.

## Data

Train/test split files are not included. Place the released files under `datasets/` with the names listed in [`datasets/readme.md`](datasets/readme.md) (`PLV_train.tsv`, `PLV_test.tsv`, `AW_train.tsv`, `AW_test.tsv`). Run all commands from this directory so that relative paths resolve.

## Main Experiments

PLV root-level coding (Table 2, Table 3, Table 5):

```bash
python3 primary/plover_predictive_experiments.py --step llm_no_cb
python3 primary/plover_predictive_experiments.py --step llm_compact
python3 primary/plover_predictive_experiments.py --step llm_enriched
python3 primary/plover_predictive_experiments.py --step llm_icl
python3 primary/plover_predictive_experiments.py --step llm_cot
python3 primary/plover_predictive_experiments.py --step table
```

AW binary coding (Table 2, Table 3):

```bash
python3 primary/aw_predictive_experiments.py --step llm_no_cb
python3 primary/aw_predictive_experiments.py --step llm_compact
python3 primary/aw_predictive_experiments.py --step llm_enriched
python3 primary/aw_predictive_experiments.py --step llm_icl
python3 primary/aw_predictive_experiments.py --step llm_cot_cb
python3 primary/aw_predictive_experiments.py --step table
```

Add `--limit 5` for a quick test run. Predictions are written to `outputs/`, and `--step table` summarizes the completed runs.

## Supplemental Experiments

| Script | Purpose | Paper |
|---|---|---|
| `supplemental/rag/plover_rag.py --step rag_cb` | RAG over the Enriched codebook, PLV | Table 3, Appendix B |
| `supplemental/rag/aw_rag_experiments.py --step rag_cb` | RAG over the Enriched codebook, AW | Table 3 |
| `supplemental/plover_binary_quad_experiments.py` | Direct binary and quad-level PLV classification (Gemma2:9B) | Table 7 |
| `primary/plover_predictive_experiments.py --step tree` (also `tiny`, `full`) | NLI baselines of Hu et al. (2024), using `prompts/` and `legacy_nli/` | Table 3 |

`supplemental/rag/*_rag_v2_experiments.py` is an earlier retrieval variant that is not reported in the paper. `tools/parse_cameo_to_json.py` is an optional utility for converting the CAMEO manual to JSON.

## Directory Structure

```text
primary/        main PLV and AW predictive scripts
supplemental/   RAG and direct binary/quad experiments
legacy_nli/     NLI zero-shot baseline used by the tree/tiny/full steps
prompts/        NLI hypothesis templates (Tiny, Full, Tree)
codebooks/      PLOVER and CAMEO manuals (PDF)
datasets/       location for the released split files
tools/          optional conversion utilities
```

Model outputs, score arrays, figures, and split files are not included.
