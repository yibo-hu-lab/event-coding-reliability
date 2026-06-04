# Part II: Behavioral Probes

This folder contains the simplified code for the Part II behavioral checks for codebook-grounded source-target event coding. It is separate from `../part1_predictive_evaluation/` because these checks use Hugging Face models, while the Part I prediction scripts use Ollama. Large datasets, result files, notebooks, saved predictions, and editor files are not included.

## Contents

- `run.py`: command-line runner.
- `code/`: main code for evaluation, summaries, and split creation.
- `data/codebooks/`: lightweight codebook files used by the behavioral and predictive methods.
- `data/prompt_assets/`: few-shot and RAG prompt files.
- `data/raw_data/`: normalized raw CSV files used to rebuild splits.
- `data/dataset_splits/`: place train/dev/test split CSV files here.
- `examples/toy_plover_splits/`: tiny synthetic split files for checking the command line.
- `results/`: result folder created when commands are run.

## Paper Alignment

The default behavioral run matches the Part II diagnostics reported in the paper:

- original-condition accuracy under the unmodified codebook
- legal-label compliance and definition recovery, summarized as `CB-Align.`
- order perturbations with reversed and shuffled codebooks
- generic-label probes
- swapped label-definition mapping probes
- `Rule-S`, computed as the mean of order agreement, generic-label F1, and swapped-mapping F1

The included `plover_enriched_codebook.txt` and `aw_enriched_codebook.txt` files are the enriched codebooks with definitions, examples, event-mode guidance, and boundary notes. Compact versions are also included as `plover_compact_codebook.txt` and `aw_compact_codebook.txt`.

## Data Format

To run evaluation directly, place split files under `data/dataset_splits/`:

```text
<dataset>_train.csv
<dataset>_dev.csv
<dataset>_test.csv
```

Required columns:

```text
text,label
```

Optional columns used when available:

```text
meta,context,source,target
```

To rebuild splits from raw data, use the normalized CSVs under `data/raw_data/` and run the split builder. The dataset slugs are `plover` and `aw`; the PLOVER raw file is named `plv.csv` for compatibility with the source export.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU-backed model runs require a compatible PyTorch/CUDA installation and access to the selected Hugging Face model.

## Smoke Test

The following commands use synthetic toy data, so they are not paper results. They only confirm that the code can read splits and codebooks.

```bash
mkdir -p data/dataset_splits
cp examples/toy_plover_splits/*.csv data/dataset_splits/
python run.py --help
python run.py --dataset-stats --stats-datasets plover
```

The stats command writes outputs under:

```text
results/dataset_stats/
```

## Reproducing Runs

Generate dataset splits:

```bash
python run.py --make-dataset-splits --split-builder-datasets plover,aw
```

Run behavioral probes:

```bash
python run.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --datasets plover \
  --limit 200 \
  --quantization 4
```

By default this runs all paper behavioral probes. To run a subset, add one or more of:

```bash
--behavioral-codebook     # definition recovery
--behavioral-unlabeled    # order perturbations + legal-label compliance
--behavioral-labeled      # original accuracy + generic-label + swapped-mapping probes
```

Run zero-shot evaluation only:

```bash
python run.py \
  --only-zeroshot \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --zeroshot-datasets plover \
  --zeroshot-limit 5
```

Generate dataset descriptives:

```bash
python run.py --dataset-stats --stats-datasets plover
```

## Outputs

Main outputs are written under:

```text
results/behavioral_results/
results/behavioral_prediction_cache/
results/zero_shot_results/
results/dataset_stats/
```

Behavioral runs write three summary CSVs:

```text
*_results.csv                    # long-form probe metrics
*_paper_probe_breakdown.csv      # Orig./Rev./Shuf./Order kappa/Generic F1/Swap F1
*_paper_summary.csv              # Orig. Acc., CB-Align., Rule-S
```

Result files are not included.

## Notes

- This folder does not include machine-specific paths, notebooks, saved predictions, result files, or full datasets.
- If using a custom dataset slug, pass a codebook explicitly with `--codebook-new-file` or set `BEHAVIOR_<SLUG>_CODEBOOK_NEW`.
- New result files are written under `results/` and are ignored by git.

## What Is Included

This folder includes the Part II probes, zero-shot evaluation, CoT/ICL/RAG-style prediction scripts, split creation, and descriptive summaries.

It includes normalized raw CSVs for creating splits, but it does not include large model weights, original source files, precomputed split CSVs, saved predictions, or paper figures. Behavioral runs can be rerun from the normalized raw CSVs using the split script.

For Part I prediction scripts, see `../part1_predictive_evaluation/`. Other experiment folders, notebooks, saved outputs, and plotting work folders are not included here.
