"""Subprocess wrapper for behavioral_tests.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIR = ROOT / "code"
BEHAVIORAL_RESULTS_DIR = ROOT / "results" / "behavioral_results"


def run_behavioral_tests_subprocess(
    model_name: str,
    datasets: List[str],
    limit: int,
    quantization: str = "4",
    env: Optional[Dict[str, str]] = None,
    probe_extras: Optional[Sequence[str]] = None,
) -> None:
    cmd = [
        sys.executable,
        str(CODE_DIR / "evaluation" / "behavioral_tests.py"),
        "--model-name",
        model_name,
        "--quantization",
        quantization,
        "--limit",
        str(limit),
        "--output-dir",
        str(BEHAVIORAL_RESULTS_DIR),
    ]
    for ds in datasets:
        cmd.extend(["--datasets", ds])
    if probe_extras:
        cmd.extend(list(probe_extras))
    print("[run] ", " ".join(cmd))
    _env = dict(os.environ if env is None else env)
    existing = _env.get("PYTHONPATH", "")
    code_path = str(CODE_DIR)
    _env["PYTHONPATH"] = code_path if not existing else code_path + os.pathsep + existing
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log: list[str] = []
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                log.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        ret = proc.wait(timeout=None)
    if ret != 0:
        tail = "".join(log)
        print(
            "\n----- behavioral_tests.py failed (exit %s); tail of output -----\n" % ret,
            file=sys.stderr,
        )
        print(tail[-20000:] if len(tail) > 20000 else tail, file=sys.stderr)
        print("----- end -----\n", file=sys.stderr)
        raise subprocess.CalledProcessError(ret, cmd)
