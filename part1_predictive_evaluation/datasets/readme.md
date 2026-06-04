# Dataset Files

Precomputed train/test split files are not included. This keeps the Part I directory focused on prompt and experiment logic, avoids shipping restricted split files, and avoids ambiguity about whether a copied split differs from the split used in the paper.

To rerun the Part I predictive scripts, place the released split files here with these names:

```text
PLV_train.tsv
PLV_test.tsv
AW_train.tsv
AW_test.tsv
```

For the Part II behavioral reliability code, use `../part2_behavioral_probes/`, which includes normalized raw CSV files and a split script.
