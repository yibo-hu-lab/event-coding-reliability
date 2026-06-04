"""Stage 1 (behavioral_tests) + optional Stage 2 (zero-shot). Invoked from run.py / notebooks."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent


def _env_run_behavioral_in_process() -> bool:
    for k in ("BEHAVIOR_IN_PROCESS", "BEHAVIOR_BEHAVIORAL_IN_PROCESS", "DATAVERSE_BEHAVIORAL_IN_PROCESS"):
        if os.environ.get(k, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


_DATASET_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


def _validate_dataset_slugs(names: Sequence[str], parser: argparse.ArgumentParser) -> None:
    for raw in names:
        d = raw.strip().lower()
        if len(d) >= 64 or not _DATASET_SLUG_RE.fullmatch(d):
            parser.error(
                f"Invalid dataset slug {raw!r}. "
                "Use lowercase letters/digits plus optional '-' or '_' (max 63 chars)."
            )


def _merge_overlay_into_child_env(child_env: dict, chosen: list[str], args: argparse.Namespace) -> None:
    """Wire --codebook-new-file / --raw-data-file when exactly one slug is chosen."""
    if len(chosen) != 1:
        return
    key = chosen[0].upper()
    if args.codebook_new_file is not None:
        resolved = str(Path(args.codebook_new_file).resolve())
        child_env[f"BEHAVIOR_{key}_CODEBOOK_NEW"] = resolved
        child_env[f"DATAVERSE_{key}_CODEBOOK_NEW"] = resolved
    if args.raw_data_file is not None:
        resolved = str(Path(args.raw_data_file).resolve())
        child_env[f"BEHAVIOR_{key}_RAW_DATA_FILE"] = resolved
        child_env[f"DATAVERSE_{key}_RAW_DATA_FILE"] = resolved


def _resolve_codebook_paths_under_environ(dataset: str, environ: Dict[str, str]) -> tuple[Path, Path]:
    """Resolve structured codebook path as downstream code sees it."""
    saved = dict(os.environ)
    try:
        merged = dict(saved)
        merged.update({str(k): str(v) for k, v in environ.items()})
        os.environ.clear()
        os.environ.update(merged)

        from core.codebook_utils import resolve_codebook_paths

        return resolve_codebook_paths(dataset.lower())
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _split_paths(dataset: str) -> tuple[Path, Path, Path]:
    splits = ROOT / "data" / "dataset_splits"
    return (
        splits / f"{dataset}_train.csv",
        splits / f"{dataset}_dev.csv",
        splits / f"{dataset}_test.csv",
    )


def _has_complete_splits(dataset: str) -> bool:
    return all(path.is_file() for path in _split_paths(dataset))


def _materialize_missing_splits(dataset: str, args: argparse.Namespace, splits: Path, chosen: list[str]) -> None:
    from data_prep.make_dataset_splits import write_splits

    child_env = os.environ.copy()
    _merge_overlay_into_child_env(child_env, chosen, args)
    os.environ.update(child_env)
    train_path, dev_path, test_path = write_splits(dataset, output_dir=splits)
    print(
        f"[run] Materialized {dataset} splits from raw data:\n"
        f"  {train_path}\n  {dev_path}\n  {test_path}"
    )


def _run_zeroshot_subprocess(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required for zero-shot runs.")
    cmd = [
        sys.executable,
        str(ROOT / "code" / "evaluation" / "run_zero_shot.py"),
        "--model-name",
        args.model,
        "--datasets",
        args.zeroshot_datasets,
        "--limit",
        str(args.zeroshot_limit),
        "--split",
        "dev",
        "--codebook-type-list",
        "new_format",
    ]
    print("[run] zero-shot (run_zero_shot.py):", " ".join(cmd))
    env = os.environ.copy()
    zs = (args.zeroshot_datasets or "").strip()
    zs_slugs = [s.strip().lower() for s in zs.split(",") if s.strip()]
    # Same env wiring as behavioral: `--codebook-new-file` / `--raw-data-file` affect one slug only.
    if len(zs_slugs) == 1:
        _merge_overlay_into_child_env(env, zs_slugs, args)
    code_path = str(ROOT / "code")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = code_path if not existing else code_path + os.pathsep + existing
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def _run_script_subprocess(
    script_name: str, script_args: list[str], label: str, env: Dict[str, str] | None = None
) -> None:
    script_map = {
        "make_dataset_splits.py": ROOT / "code" / "data_prep" / "make_dataset_splits.py",
        "get_dataset_stats.py": ROOT / "code" / "reporting" / "get_dataset_stats.py",
    }
    script_path = script_map[script_name]
    cmd = [sys.executable, str(script_path), *script_args]
    print(f"[run] {label}:", " ".join(cmd))
    run_env = dict(os.environ if env is None else env)
    code_path = str(ROOT / "code")
    existing = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = code_path if not existing else code_path + os.pathsep + existing
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=run_env)


def _selected_utility_modes(args: argparse.Namespace) -> list[str]:
    modes = []
    if args.make_dataset_splits:
        modes.append("make_dataset_splits")
    if args.dataset_stats:
        modes.append("dataset_stats")
    return modes


def _execute_utility_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> bool:
    modes = _selected_utility_modes(args)
    if not modes:
        return False
    if len(modes) > 1:
        parser.error(f"Select only one utility mode at a time, got: {modes}")

    mode = modes[0]
    if mode == "make_dataset_splits":
        spec = (args.split_builder_datasets or "").strip()
        if not spec:
            parser.error("--split-builder-datasets is required with --make-dataset-splits (comma-separated slugs).")
        slugs = [s.strip().lower() for s in spec.split(",") if s.strip()]
        _validate_dataset_slugs(slugs, parser)
        if (args.raw_data_file is not None or args.codebook_new_file is not None) and len(slugs) != 1:
            parser.error(
                "--raw-data-file/--codebook-new-file require exactly one slug in --split-builder-datasets."
            )
        child_env = os.environ.copy()
        _merge_overlay_into_child_env(child_env, slugs, args)
        cmd = ["--datasets", ",".join(slugs)]
        _run_script_subprocess("make_dataset_splits.py", cmd, "dataset splits", env=child_env)
        return True

    if mode == "dataset_stats":
        spec = (args.stats_datasets or "").strip()
        if not spec:
            spec = "auto"
        cmd = ["--datasets", spec, "--tokenizer-name", args.stats_tokenizer_name]
        if args.stats_write_latex:
            cmd.append("--write-latex")
        _run_script_subprocess("get_dataset_stats.py", cmd, "dataset stats")
        return True

    parser.error(f"Unsupported utility mode: {mode}")
    return False


def build_parser() -> argparse.ArgumentParser:
    epilog = "Behavioral probes and optional zero-shot evaluation are separate. Utility modes (--make-dataset-splits, --dataset-stats) run one script at a time."
    p = argparse.ArgumentParser(
        description="Paper behavioral reliability probes; optional zero-shot evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id. Required for behavioral and zero-shot modes.",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Lowercase slug(s) for each dataset (repo data/... splits + codebook resolution). "
        "With exactly one slug, use --codebook-new-file / --raw-data-file to set paths. "
        "Required for behavioral tests unless --only-zeroshot.",
    )
    p.add_argument("--quantization", default="4")
    p.add_argument("--limit", type=int, default=200, help="Sample cap for behavioral_tests")

    p.add_argument(
        "--codebook-new-file",
        type=Path,
        default=None,
        help=(
            "When exactly one slug is listed in --datasets or --split-builder-datasets, set "
            "`BEHAVIOR_<SLUG_UPPER>_CODEBOOK_NEW` (and legacy `DATAVERSE_*`) to this file path."
        ),
    )
    p.add_argument(
        "--raw-data-file",
        type=Path,
        default=None,
        help=(
            "When exactly one slug is listed in --datasets or --split-builder-datasets, set "
            "`BEHAVIOR_<SLUG_UPPER>_RAW_DATA_FILE` (and legacy `DATAVERSE_*`)."
        ),
    )

    p.add_argument("--only-zeroshot", action="store_true", help="Only run run_zero_shot.py")
    p.add_argument("--zeroshot", action="store_true", help="After behavioral tests, run run_zero_shot.py")
    p.add_argument(
        "--zeroshot-datasets",
        type=str,
        default="",
        help="Comma-separated slug(s) passed to run_zero_shot.py when using --only-zeroshot / --zeroshot.",
    )
    p.add_argument("--zeroshot-limit", type=int, default=5)
    p.add_argument(
        "--behavioral-in-process",
        action="store_true",
        help="Run behavioral_tests in-process (Jupyter tracebacks). "
        "Env: BEHAVIOR_IN_PROCESS=1 or legacy DATAVERSE_BEHAVIORAL_IN_PROCESS=1.",
    )
    p.add_argument(
        "--behavioral-codebook",
        action="store_true",
        help="Pass --codebook to behavioral_tests (definition recovery only).",
    )
    p.add_argument(
        "--behavioral-unlabeled",
        action="store_true",
        help="Forward --unlabeled to behavioral_tests (order perturbations + legal-label compliance).",
    )
    p.add_argument(
        "--behavioral-labeled",
        action="store_true",
        help="Forward --labeled to behavioral_tests (original accuracy + generic/swap probes).",
    )

    p.add_argument(
        "--make-dataset-splits",
        action="store_true",
        help="Run code/data_prep/make_dataset_splits.py (train/dev/test CSVs).",
    )
    p.add_argument(
        "--split-builder-datasets",
        type=str,
        default="",
        help="For --make-dataset-splits: comma-separated slug(s) (required with that flag).",
    )
    p.add_argument(
        "--dataset-stats",
        action="store_true",
        help="Run code/reporting/get_dataset_stats.py",
    )
    p.add_argument(
        "--stats-datasets",
        type=str,
        default="",
        help="Comma-separated slug(s) for --dataset-stats, or omit/empty/'auto' to use every slug with "
        "complete train/dev/test splits under data/dataset_splits/.",
    )
    p.add_argument(
        "--stats-tokenizer-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer used by --dataset-stats for codebook length estimates.",
    )
    p.add_argument("--stats-write-latex", action="store_true", help="Also write LaTeX output for --dataset-stats")
    return p


def _behavioral_probe_cli_extras(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.behavioral_codebook:
        out.append("--codebook")
    if args.behavioral_unlabeled:
        out.append("--unlabeled")
    if args.behavioral_labeled:
        out.append("--labeled")
    return out


def _run_behavioral_tests_in_process(
    model: str,
    datasets: List[str],
    limit: int,
    quantization: str,
    child_env: dict,
    probe_extras: List[str],
) -> None:
    """Invoke run_tests like the subprocess path."""
    code_dir = str(ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    saved_env = dict(os.environ)
    old_cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        from click.testing import CliRunner

        # Notebook kernels cache imports; a stale `behavioral_tests` breaks after on-disk edits.
        for _name in ("behavioral_tests",):
            sys.modules.pop(_name, None)
        from evaluation.behavioral_tests import run_tests

        cli_args = [
            "--model-name",
            model,
            "--quantization",
            str(quantization),
            "--limit",
            str(limit),
            "--output-dir",
            str(ROOT / "results" / "behavioral_results"),
        ]
        for ds in datasets:
            cli_args.extend(["--datasets", ds])
        cli_args.extend(probe_extras)
        print("[run] behavioral_tests (in-process):", " ".join(cli_args))
        # CliRunner isolation can miss overrides; prime os.environ so path env vars behave.
        # os.environ (then restore in outer finally) so custom slugs match preflight resolution.
        primed = {str(k): str(v) for k, v in child_env.items()}
        os.environ.clear()
        os.environ.update(primed)
        runner = CliRunner()
        result = runner.invoke(run_tests, cli_args, catch_exceptions=True)
        if result.exit_code != 0:
            out = (result.output or "").strip()
            if out:
                print(out, file=sys.stderr)
            if result.exception is not None:
                raise result.exception
            raise RuntimeError(f"behavioral_tests exited with code {result.exit_code}")
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(saved_env)


def execute_pipeline(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Main entry; parser.error for bad CLI."""
    if _execute_utility_mode(args, parser):
        print("[run] Done.")
        return

    if args.only_zeroshot and args.zeroshot:
        parser.error("Use either --only-zeroshot OR --zeroshot, not both.")

    if args.only_zeroshot:
        if not args.model:
            parser.error("--model is required with --only-zeroshot.")
        if args.datasets:
            parser.error("With --only-zeroshot, do not pass --datasets.")
        zs = (args.zeroshot_datasets or "").strip()
        if not zs:
            parser.error("--zeroshot-datasets is required with --only-zeroshot (comma-separated slugs).")
        _run_zeroshot_subprocess(args)
        print("[run] Done (zero-shot only).")
        return

    if not args.datasets:
        parser.error("--datasets is required for behavioral tests (or use --only-zeroshot).")
    if not args.model:
        parser.error("--model is required for behavioral tests.")

    chosen = [x.strip().lower() for x in args.datasets]
    _validate_dataset_slugs(chosen, parser)
    if len(chosen) != len(set(chosen)):
        parser.error("--datasets contains duplicates")

    if (args.codebook_new_file is not None or args.raw_data_file is not None) and len(chosen) != 1:
        parser.error("--codebook-new-file/--raw-data-file require exactly one dataset in --datasets.")

    splits = ROOT / "data" / "dataset_splits"
    splits.mkdir(parents=True, exist_ok=True)

    for d in chosen:
        if _has_complete_splits(d):
            print(f"[run] Using existing {d} train/dev/test splits.")
            continue
        try:
            _materialize_missing_splits(d, args, splits, chosen)
        except Exception as exc:
            train, dev, test = _split_paths(d)
            parser.error(
                f"Selected {d!r} but split files are incomplete and raw-data rebuild failed.\n"
                f"Expected split files:\n  {train}\n  {dev}\n  {test}\n"
                f"Reason: {exc}"
            )
        if not _has_complete_splits(d):
            train, dev, test = _split_paths(d)
            parser.error(
                f"Selected {d!r} but split rebuild did not create the expected files:\n"
                f"  {train}\n  {dev}\n  {test}"
            )

    child_env = os.environ.copy()
    _merge_overlay_into_child_env(child_env, chosen, args)

    for d in chosen:
        try:
            new_p, _old = _resolve_codebook_paths_under_environ(d, child_env)
        except ValueError as exc:
            parser.error(f"Dataset {d!r}: invalid codebook resolution: {exc}")
        if not new_p.is_file():
            dsu = d.upper()
            parser.error(
                f"Missing structured codebook file for dataset {d!r}.\n"
                f"Resolved path was: {new_p}\n"
                f"Provide it via `--codebook-new-file`, `BEHAVIOR_{dsu}_CODEBOOK_NEW`, "
                f"shared `BEHAVIOR_CODEBOOK_NEW`, or legacy `DATAVERSE_*` with the same meaning."
            )

    print("[run] Behavioral probes: behavioral_tests.py.")
    in_process = args.behavioral_in_process or _env_run_behavioral_in_process()
    probe_extras = _behavioral_probe_cli_extras(args)
    if in_process:
        _run_behavioral_tests_in_process(
            args.model, chosen, args.limit, args.quantization, child_env, probe_extras
        )
    else:
        from pipeline.behavioral_subprocess import run_behavioral_tests_subprocess

        run_behavioral_tests_subprocess(
            args.model,
            chosen,
            args.limit,
            args.quantization,
            env=child_env,
            probe_extras=probe_extras,
        )

    if args.zeroshot:
        zs = (args.zeroshot_datasets or "").strip()
        if not zs:
            parser.error("--zeroshot-datasets is required when passing --zeroshot.")
        print("[run] Optional zero-shot: run_zero_shot.py.")
        _run_zeroshot_subprocess(args)

    print("[run] Done.")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: parse argv and run pipeline."""
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(list(argv))
    execute_pipeline(args, parser)


if __name__ == "__main__":
    main()
