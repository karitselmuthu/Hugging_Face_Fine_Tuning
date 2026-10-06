# Emotion validation error review

This review uses the saved `balanced_25_validation_evaluation.json` from
`smollm2-360m-lora-2000`. It contains 25 validation examples for each label.
The reference labels are the dataset's labels; some short statements can be
ambiguous even when the model output differs from the reference.

| Reference label | Correct | Predicted joy | Predicted sadness | Predicted fear | Other wrong labels |
| --- | ---: | ---: | ---: | ---: | --- |
| love | 11 | 10 | 3 | 0 | anger: 1 |
| surprise | 10 | 10 | 0 | 4 | love: 1 |

The model frequently maps `love` to the broader positive label `joy`,
including references about affection, loyalty, and gratitude. It also maps
many `surprise` examples to `joy`, including references about curiosity or
amazement. Four `surprise` examples become `fear`, where the wording describes
an unusual or unsettling feeling. These are observed confusions in a small,
balanced validation slice, not a claim that every individual dataset label is
unambiguous.

The original 2,000-example training subset, shuffled with seed 42, contained:

| Label | Original subset | Equal-count subset |
| --- | ---: | ---: |
| sadness | 591 | 334 |
| joy | 639 | 334 |
| love | 174 | 333 |
| anger | 264 | 333 |
| fear | 252 | 333 |
| surprise | 80 | 333 |

The full 16,000-row training split has 1,304 `love` and 572 `surprise`
examples, so an equal-count 2,000-example subset can use unique rows without
duplicating either rare label. The next controlled experiment changes only
the training subset composition: keep SmolLM2-360M, LoRA settings, seed,
2,000 training examples, 64 monitored validation examples, and one epoch.
The `--balanced-train` option draws 333 or 334 rows per label without
replacement and records the counts in `run_summary.json`. Validation
selection remains unchanged.

Compare the new adapter with `smollm2-360m-lora-2000` on the same balanced
validation slice. Also compare them on the same ordinary 128-example
validation sample to detect a possible cost to common-label accuracy. A gain
in rare-label recall alone is insufficient if common-label performance falls
too far for the intended use. Use validation results to choose the setup;
the previously inspected test cases should not guide the training choice.

The sampling code passed a deterministic 2,000-example selection check and
a two-step CPU training check. The complete balanced SmolLM2-360M MPS run
then finished in 434.3 seconds. On the fixed 150-example balanced validation
slice, accuracy rose from 60.7% to 70.7% and macro-F1 from 0.598 to 0.707.
`Love` and `surprise` each rose to 19/25 correct, while `joy` recall fell
from 24/25 to 20/25. On the matching ordinary 128-example validation
sample, the original shuffled adapter scored 73.4% accuracy and 0.670
macro-F1; the equal-count adapter scored 70.3% and 0.612. Equal-count
training improved `love` recall but reduced `joy` recall. The choice
depends on whether the application values overall accuracy under the
original label mix or more equal treatment of rare labels.
