# Part II: Rule-Adherence Probes

This directory contains the code for the rule-adherence probes in the paper. The probes ask whether model predictions follow the codebook when the codebook is changed in controlled ways. Unlike the Part I scripts, which query models through Ollama, these probes load models from Hugging Face. The predictive experiments are in [`../part1_predictive_evaluation/`](../part1_predictive_evaluation/).

## Probes

Each probe reruns the same inputs under a modified codebook and compares the predictions with those under the original codebook.

| Probe | Codebook change | Expected behavior if the model follows the codebook | Metric |
|---|---|---|---|
| Original | None | Baseline accuracy | Accuracy |
| Order | Entries reversed or shuffled; labels and definitions unchanged | Predictions remain stable | Fleiss' κ across original, reversed, and shuffled orders |
| Generic labels | Label names replaced with `LABEL_1`, `LABEL_2`, ... | Predictions follow the definitions | Weighted F1 against remapped gold labels |
| Swapped mapping | Label names reassigned by a derangement, so no definition keeps its original name | Predictions follow the definitions, not the familiar names | Weighted F1 against remapped gold labels |

Results are reported in Table 4 and Table 8 (PLV), Table 9 (AW), and Table 10 (a swapped-mapping example). All probe runs use greedy decoding.

## Models

The paper uses four instruction-tuned models, loaded with 4-bit quantization:

```text
google/gemma-2-9b-it
Qwen/Qwen2.5-7B-Instruct
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Access to the Gemma, Mistral, and Llama models requires accepting their licenses on Hugging Face. GPU runs require a compatible PyTorch/CUDA installation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

`data/raw_data/` contains normalized PLV (`plv.csv`) and AW (`aw.csv`) rows from Hu et al. (2024). Build the train/dev/test splits with:

```bash
python run.py --make-dataset-splits --split-builder-datasets plover --raw-data-file data/raw_data/plv.csv
python run.py --make-dataset-splits --split-builder-datasets aw
```

The splits are written to `data/dataset_splits/` with a fixed seed (42). The probes run on the dev splits, which contain 312 PLV and 253 AW examples, matching the evaluation subsets in the paper. Split files contain the columns `text,label`, with optional `meta,context,source,target`.

## Running the Probes

To reproduce a paper run, use the full dev split (`--limit 0`) and 4-bit quantization:

```bash
python run.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --datasets plover \
  --limit 0 \
  --quantization 4
```

Replace `--datasets plover` with `--datasets aw` for AW, and repeat for each model. The default `--limit 200` samples 200 dev examples, which is useful for quicker runs.

To run a subset of probes:

```bash
--behavioral-labeled      # original, generic-label, and swapped-mapping probes
--behavioral-unlabeled    # order probes
```

With neither flag, all probes run, together with a definition-recovery check (`--behavioral-codebook`) that is not reported in the paper.

## Codebooks and Prompt Assets

```text
data/codebooks/plover_compact_codebook.txt    PLV, Compact
data/codebooks/plover_enriched_codebook.txt   PLV, Enriched
data/codebooks/aw_compact_codebook.txt        AW, Compact
data/codebooks/aw_enriched_codebook.txt       AW, Enriched
data/prompt_assets/                           ICL demonstrations and RAG retrieval bundles
```

The Compact codebook contains label names, short definitions, and quad-level groupings. The Enriched codebook adds examples, event-mode guidance, boundary notes, and disambiguation rules. The codebooks are adapted from PLOVER (CC BY-SA 4.0).

The ICL, CoT, and RAG probe rows in Table 4 use the pipelines in `code/evaluation/` (`icl_behavior_pipeline.py`, `cot_behavior_pipeline.py`, `rag_behavior_pipeline.py`). These modules are called from Python and do not have a command-line entry point in `run.py`. `ragv2_behavior_pipeline.py` is an earlier retrieval variant that is not reported in the paper.

## Outputs

Results are written under `results/` (not tracked by git):

```text
results/behavioral_results/
  *_results.csv                  long-form probe metrics
  *_paper_probe_breakdown.csv    original, reversed, shuffled, order κ, generic F1, swap F1
  *_paper_summary.csv            original accuracy and two aggregate columns
results/behavioral_prediction_cache/
```

The aggregate columns `CB-Align.` and `Rule-S` in `*_paper_summary.csv` come from an earlier draft and are not reported in the paper.

## Other Utilities

```bash
python run.py --only-zeroshot --model <hf_model_id> --zeroshot-datasets plover --zeroshot-limit 5
python run.py --dataset-stats --stats-datasets plover
```

`examples/toy_plover_splits/` contains small synthetic splits for checking that the code can read data and codebooks. These are not paper data.

## Directory Structure

```text
run.py                    command-line runner
code/core/                probe construction, codebook parsing, model loading
code/evaluation/          probe runner, zero-shot evaluation, ICL/CoT/RAG pipelines
code/data_prep/           split builder
code/pipeline/            runner internals
code/reporting/           dataset statistics and example export
data/codebooks/           Compact and Enriched codebooks (PLV, AW)
data/prompt_assets/       ICL and RAG prompt files
data/raw_data/            normalized PLV and AW rows
examples/                 synthetic toy splits
```
