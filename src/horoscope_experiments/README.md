# Horoscope learning scripts

These scripts preserve the original step-by-step horoscope experiments.
They are separate from the reusable task pipeline in the parent `src/`
directory. Run them from the project root so their relative `data/` and
`models/` paths resolve correctly.

| Stage | Scripts |
| --- | --- |
| First model | `prepare_dataset.py`, `train.py`, `inference.py` |
| Instruction and response-only training | `train_instruction.py`, `inference_instruction.py`, `train_response_only.py`, `train_response_only_2000.py`, `inference_response_only.py` |
| Reviews | `evaluate_input_sensitivity.py`, `evaluate_heldout.py`, `analyze_training_data.py`, `evaluate_adherence.py` |
| LoRA and QLoRA | `train_lora.py`, `inference_lora.py`, `prepare_mlx_qlora.py`, `review_mlx_qlora.py` |

See the [experiment guide](../../docs/horoscope_experiments.md) for current
commands and the [learning journal](../../hugging_face_fine_tuning_learnings.md)
for the sequence of experiments. The reusable horoscope scenario is
configured separately in [tasks/horoscope.json](../../tasks/horoscope.json).
