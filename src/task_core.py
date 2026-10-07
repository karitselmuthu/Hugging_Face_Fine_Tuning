"""Compatibility imports for the earlier single-module task API.

New pipeline stages import their focused modules directly.
"""

from task_config import load_task, preparation_spec
from task_io import ROOT, prepared_dir, read_jsonl
from task_model import ResponseOnlyCollator, choose_device, load_saved_run
from task_prompts import encode_example, format_prompt

__all__ = ["ROOT", "ResponseOnlyCollator", "choose_device", "encode_example",
           "format_prompt", "load_saved_run", "load_task", "preparation_spec",
           "prepared_dir", "read_jsonl"]
