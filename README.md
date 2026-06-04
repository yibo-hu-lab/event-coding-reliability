# Simplified Code

This folder follows the two-part structure of the paper.

## Part I: Predictive Evaluation

`part1_predictive_evaluation/` contains the code for the prediction experiments:

- PLV root-level source-target event coding.
- AW binary Cooperation/Conflict coding.
- Prompt variants: `No Codebook`, `Compact`, `Enriched`, `ICL`, `CoT`, and `RAG`.
- Extra scripts for binary/quad PLV results, RAG runs, older NLI baselines, codebook conversion, and figures.

To rerun these scripts, place the released TSV splits under `part1_predictive_evaluation/datasets/`. The scripts use Ollama models by default. Tables, figures, score files, and TSV splits are not included.

## Part II: Behavioral Probes

`part2_behavioral_probes/` contains the code for the behavioral checks:

- Original-condition accuracy.
- Legal-label compliance and definition recovery, summarized as `CB-Align.`.
- Order perturbation probes.
- Generic-label probes.
- Swapped label-definition mapping probes.
- `Rule-S`, computed from order agreement, generic-label F1, and swapped-mapping F1.

This directory includes the command-line runner, normalized raw CSVs, codebooks, split script, summary scripts, and small toy data.

## What Is Included

- Part II help commands and toy-data checks can run immediately.
- Part II `plover` and `aw` runs can be started after creating splits from `part2_behavioral_probes/data/raw_data/`.
- Part I prediction scripts need the released TSV splits and the named model.
- Outputs, score files, notebooks, figures, original source files, large model weights, and editor files are not included.

## Basic Commands

```bash
cd part2_behavioral_probes
python run.py --help
```

```bash
cd part1_predictive_evaluation
python3 primary/plover_predictive_experiments.py --help
python3 primary/aw_predictive_experiments.py --help
```

Older working-folder names are not used here.
