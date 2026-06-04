# Part I: Predictive Evaluation

This directory contains the simplified code for the Part I prediction experiments. It includes the PLV and AW prediction scripts, prompt files, codebook files, and small supporting scripts for the prompt variants. The Part II behavioral checks are in `../part2_behavioral_probes/`.

The terminology here now follows the paper:

- `No Codebook`: label list only
- `Compact`: short definitions and quad-level grouping
- `Enriched`: same label space plus examples, event-mode guidance, boundary notes, and disambiguation rules
- `ICL`: labeled demonstrations without full codebook definitions
- `CoT`: enriched-style reasoning prompt before the final label
- `RAG`: retrieval from enriched codebook definitions/examples/rules

## Setup

```bash
pip install transformers torch pandas scikit-learn requests
pip install sentence-transformers faiss-cpu  # optional, for RAG
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

The scripts use Ollama by default and write result files under `outputs/` and `plots/`. Those result folders are not included. Some scripts also allow a different model name through command arguments or settings set before running.

Precomputed train/test split files are also omitted. To rerun these predictive scripts, place the released split TSV files under `datasets/` as described in `datasets/readme.md`.

## What Is Included

This directory keeps only the files needed for the simplified Part I code:

- Main predictive prompt code is included for PLV root labels and AW binary labels.
- Supplemental RAG, binary/quad PLV, NLI baseline, plotting, and conversion scripts are grouped by purpose.
- Result CSVs, figures, score arrays, and released TSV split files are not included.
- To match the reported numbers, use the released TSV splits, a running Ollama server with the named model, and the same decoding settings.

## Folder Contents

- `primary/`: main Part I predictive scripts for PLV root labels and AW binary labels.
- `supplemental/`: additional PLV binary/quad experiments.
- `supplemental/rag/`: PLV and AW RAG variants.
- `supplemental/plots/`: plotting scripts for paper-aligned predictive summaries.
- `legacy_nli/`: older NLI zero-shot prompt baseline used by the PLV scripts.
- `tools/`: one-off conversion scripts, such as CAMEO PDF parsing.
- `datasets/`: put released TSV split files here.
- `prompts/`, `codebooks/`: prompt templates and manual/codebook files used by the predictive scripts.

Run commands from this `part1_predictive_evaluation/` directory so relative paths such as `datasets/`, `outputs/`, `prompts/`, and `scores/` resolve consistently.

## Main Predictive Runs

PLV root-level source-target event coding:

```bash
python3 primary/plover_predictive_experiments.py --step llm_no_cb
python3 primary/plover_predictive_experiments.py --step llm_compact
python3 primary/plover_predictive_experiments.py --step llm_enriched
python3 primary/plover_predictive_experiments.py --step llm_icl
python3 primary/plover_predictive_experiments.py --step llm_cot
python3 primary/plover_predictive_experiments.py --step table
```

AW binary Cooperation/Conflict coding:

```bash
python3 primary/aw_predictive_experiments.py --step llm_no_cb
python3 primary/aw_predictive_experiments.py --step llm_compact
python3 primary/aw_predictive_experiments.py --step llm_enriched
python3 primary/aw_predictive_experiments.py --step llm_icl
python3 primary/aw_predictive_experiments.py --step llm_cot_cb
python3 primary/aw_predictive_experiments.py --step table
```

Use `--limit 5` for a quick smoke run.

## Supplemental Scripts

- `supplemental/plover_binary_quad_experiments.py`: direct binary and quad-level PLV classification.
- `supplemental/rag/plover_rag.py`, `supplemental/rag/plover_rag_v2_experiments.py`: PLV retrieval variants.
- `supplemental/rag/aw_rag_experiments.py`, `supplemental/rag/aw_rag_v2_experiments.py`: AW retrieval variants.
- `tools/parse_cameo_to_json.py`: optional JSON codebook conversion.
- `supplemental/plots/plot_plover_predictive_results.py`, `supplemental/plots/plot_aw_predictive_results.py`: plotting scripts using the paper-aligned method names and current table values.

## Part II Behavioral Probes

For `Orig. Acc.`, `CB-Align.`, `Rule-S`, order, generic-label, and swapped-mapping diagnostics, use the Part II folder:

```bash
cd ../part2_behavioral_probes
python run.py --help
```

Result files, scores, figures, notebooks, editor files, and precomputed split files are intentionally excluded. Re-run the commands above after placing the released split TSV files under `datasets/`.
