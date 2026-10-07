# Horoscope through the governed pipeline

These commands use the same preparation, audit, lineage, evaluation, and
inference entry points as the other tasks. Run them from the repository root.
Choose fresh output directory names if these already exist.

```bash
.venv/bin/python src/prepare_task.py --task tasks/horoscope.json --output-dir data/tasks/horoscope_governed --deduplicate-cross-split --fail-on-overlap
.venv/bin/python src/audit_task_data.py --data-dir data/tasks/horoscope_governed --fail-on-overlap
.venv/bin/python src/train_task.py --task tasks/horoscope.json --data-dir data/tasks/horoscope_governed --train-samples 32 --validation-samples 8 --max-steps 2 --save-steps 1 --output-dir models/tasks/horoscope/governed-smoke
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/horoscope/governed-smoke --test-samples 12 --generation-examples 12
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/horoscope/governed-smoke --test-samples 12 --generation-examples 12 --base-only
.venv/bin/python src/infer_task.py --run-dir models/tasks/horoscope/governed-smoke --input sign=Aries --input category=General --input date=2026/09/30 --temperature 0
```

Inspect the generated examples for sign, category, and date adherence. The
short run verifies the workflow; it is not evidence of horoscope quality.
The old `src/horoscope_experiments/` scripts preserve how the project was
learned and do not produce the governed run manifest.
