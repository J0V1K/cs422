#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_pilot.config import load_config
from legal_pilot.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structure-aware overnight pilot.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "overnight_pilot.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
