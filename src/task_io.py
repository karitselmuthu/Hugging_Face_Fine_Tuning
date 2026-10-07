"""Paths and JSONL access for prepared task data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepared_dir(task):
    return ROOT / "data" / "tasks" / task["name"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                    raise ValueError(f"{path}:{number}: expected prompt and response strings")
                yield row
