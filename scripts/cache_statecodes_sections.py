#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_pilot.config import load_config
from legal_pilot.io import ensure_dir, save_json, save_sections_jsonl
from legal_pilot.statecodes import load_statecodes_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess and cache a filtered reglab/statecodes section slice."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config whose data.hf block defines the slice to cache.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output sections JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["data"]["mode"] != "hf_statecodes":
        raise ValueError("cache_statecodes_sections.py requires data.mode: hf_statecodes")

    sections = load_statecodes_sections(config)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_sections_jsonl(output_path, sections)
    save_json(
        output_path.with_suffix(".meta.json"),
        {
            "num_sections": len(sections),
            "source_config": str(Path(args.config).resolve()),
            "output_path": str(output_path.resolve()),
        },
    )
    print(f"cached_sections={len(sections)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
