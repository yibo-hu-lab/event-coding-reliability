#!/usr/bin/env python3
"""Thin CLI wrapper that forwards to the pipeline entry point."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "code"))

from pipeline.run_pipeline import main

if __name__ == "__main__":
    main()
