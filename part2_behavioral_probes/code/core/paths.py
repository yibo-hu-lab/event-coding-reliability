import os
from pathlib import Path


def find_project_root() -> Path:
    """
    Repo root = parent of `code/` when the project contains `data/codebooks`.
    Avoids matching a stray /content/data/codebooks on Colab when only code/ was copied.
    """
    env = os.environ.get("BEHAVIOR_PROJECT_ROOT", "").strip() or os.environ.get("DATAVERSE_PROJECT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "data" / "codebooks").is_dir():
            return p
    code_dir = Path(__file__).resolve().parent.parent
    candidate = code_dir.parent
    if (candidate / "data" / "codebooks").is_dir():
        return candidate
    current_dir = code_dir
    while current_dir != current_dir.parent:
        if (current_dir / "data" / "codebooks").is_dir():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(
        "Could not find project root (no data/codebooks). Set BEHAVIOR_PROJECT_ROOT or "
        "DATAVERSE_PROJECT_ROOT (legacy) to the folder that contains both code/ and data/ "
        "(same as notebook ROOT)."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
CODEBOOK_DIR = DATA_DIR / "codebooks"
# Single-level layout: keep legacy aliases so older imports still work.
CODEBOOK_NEW_DIR = CODEBOOK_DIR
CODEBOOK_ORIGINAL_DIR = CODEBOOK_DIR
DATASET_SPLITS_DIR = DATA_DIR / "dataset_splits"
RESULTS_DIR = PROJECT_ROOT / "results"
# JSONL caches for behavioral probes (dev split, keyed by model/quant/limit/variant).
BEHAVIORAL_PREDICTION_CACHE_DIR = RESULTS_DIR / "behavioral_prediction_cache"
PREDICTIONS_DIR = BEHAVIORAL_PREDICTION_CACHE_DIR  # legacy alias; same path
BEHAVIORAL_RESULTS = RESULTS_DIR / "behavioral_results"
DATASET_STATS_DIR = RESULTS_DIR / "dataset_stats"
ZERO_SHOT_RESULTS = RESULTS_DIR / "zero_shot_results"

for path in (
    DATA_DIR,
    CODEBOOK_DIR,
    CODEBOOK_NEW_DIR,
    CODEBOOK_ORIGINAL_DIR,
    DATASET_SPLITS_DIR,
    RESULTS_DIR,
):
    path.mkdir(parents=True, exist_ok=True)

for _sub in (
    BEHAVIORAL_PREDICTION_CACHE_DIR,
    BEHAVIORAL_RESULTS,
    DATASET_STATS_DIR,
    ZERO_SHOT_RESULTS,
    RAW_DATA_DIR,
):
    _sub.mkdir(parents=True, exist_ok=True)
