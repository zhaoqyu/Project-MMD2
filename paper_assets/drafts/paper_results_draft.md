# Paper Results Draft

## Dataset Analysis
The available analysis files contain 15968 example rows across detected task/split combinations.
For Task 1, the observed label distribution is: Ambivalent (3264), Clear Reply (1688), Clear Non-Reply (568), Ambivalent (1648), Clear Reply (632), Clear Non-Reply (184).
For Task 2, the observed label distribution is: Explicit (1688), Dodging (1112), Implicit (786), Deflection (619), General (616), Declining to answer (230), Claims ignorance (192), Clarification (146), Partial/half-answer (131), Explicit (616), Implicit (456), Dodging (400), General (384), Deflection (136), Declining to answer (88), Claims ignorance (64), Clarification (32), Partial/half-answer (24).

## Experimental Setup
We evaluate clarity classification and fine-grained evasion-strategy classification using macro-F1 as the primary metric. The notebook recomputes metrics from saved prediction files and uses bootstrap confidence intervals over examples.

## Main Results
The best detected TASK1 system is `DeBERTa-v3-large` on split `dev`, with macro-F1 0.690 (95% bootstrap CI 0.644-0.728).
The best detected TASK2 system is `RoBERTa-large` on split `dev`, with macro-F1 0.375 (95% bootstrap CI 0.332-0.408).

## Ablation Study
For TASK1, the strongest detected ablation family was `deberta_v3_large` with best macro-F1 0.690.
For TASK2, the strongest detected ablation family was `roberta_large` with best macro-F1 0.375.

## Error Analysis
The most frequent TASK1 confusion is gold `Ambivalent` predicted as `Clear Reply` (47 examples).
The most frequent TASK2 confusion is gold `Implicit` predicted as `Explicit` (26 examples).
Qualitative examples are saved under `paper_assets/examples/`.

## Annotator Disagreement
For Task 2, `RoBERTa-large` achieved majority-consensus macro-F1 0.342, unanimous-only macro-F1 0.402, and any-annotator match rate 0.526.

## Limitations
This analysis is limited to saved files available in the repository at notebook execution time. Missing experiments are not reported, and approximate truncation analysis uses whitespace tokenization unless tokenizer-specific lengths are added.