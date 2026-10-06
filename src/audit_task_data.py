"""Audit prepared task files without loading a model or downloading data."""

import argparse
import json
from pathlib import Path

from data_integrity import prompt_overlap, verify_prepared_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()
    artifacts = verify_prepared_data(args.data_dir)
    overlap = prompt_overlap(args.data_dir)
    print(json.dumps({"data_dir": str(args.data_dir), "artifacts": artifacts,
                      "prompt_overlap": overlap}, indent=2))
    if args.fail_on_overlap and any(overlap.values()):
        parser.exit(2, "Cross-split prompt overlap detected\n")


if __name__ == "__main__":
    main()
