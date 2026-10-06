# Local model artifacts

Training writes checkpoints, LoRA adapters, and tokenizers here. Model files are excluded from Git because they are large and may have separate redistribution terms.

Reusable runs are written to `models/tasks/<task_name>/<run_name>/`. The run directory contains the saved model or adapter, a task snapshot, and a run summary. Recreate one with `src/train_task.py`.
