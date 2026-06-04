# Dataset Splits

Place split CSV files here to run evaluation directly:

```text
<dataset>_train.csv
<dataset>_dev.csv
<dataset>_test.csv
```

For the included raw data, these files can be made from `data/raw_data/`:

```bash
python run.py --make-dataset-splits --split-builder-datasets plover,aw
```

This writes `plover_train/dev/test.csv` and `aw_train/dev/test.csv`.

Required columns:

```text
text,label
```

Optional columns:

```text
meta,context,source,target
```
