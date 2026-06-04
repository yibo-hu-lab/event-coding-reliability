# Raw Data

This folder contains normalized raw CSV files for rebuilding train/dev/test splits. Keeping raw CSVs here, rather than shipping precomputed split CSVs, makes the split procedure clear.

Files:

- `plv.csv`: normalized PLOVER-style rows used with the `plover` dataset slug.
- `aw.csv`: normalized AW-style binary Cooperation/Conflict rows used with the `aw` dataset slug.

Expected columns:

```text
text,label
```

Optional columns:

```text
meta,context,source,target
```

Build splits from `part2_behavioral_probes/` with:

```bash
python run.py --make-dataset-splits --split-builder-datasets plover,aw
```
