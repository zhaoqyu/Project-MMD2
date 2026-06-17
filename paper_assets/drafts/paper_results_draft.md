# Paper Results Draft

## Dataset Analysis
The available analysis files contain 15968 example rows across detected task/split combinations.
For Task 1, the observed label distribution is: Ambivalent (2856), Clear Reply (1477), Clear Non-Reply (497), Ambivalent (1442), Clear Reply (553), Clear Non-Reply (161).
For Task 2, the observed label distribution is: Explicit (1688), Dodging (1112), Implicit (786), Deflection (619), General (616), Ambivalent (408), Declining to answer (230), Clear Reply (211), Claims ignorance (192), Clarification (146), Partial/half-answer (131), Clear Non-Reply (71), Explicit (616), Implicit (456), Dodging (400), General (384), Ambivalent (206), Deflection (136), Declining to answer (88), Clear Reply (79), Claims ignorance (64), Clarification (32), Partial/half-answer (24), Clear Non-Reply (23).

## Experimental Setup
We evaluate clarity classification and fine-grained evasion-strategy classification using macro-F1 as the primary metric. The notebook recomputes metrics from saved prediction files and uses bootstrap confidence intervals over examples.

## Main Results
The best detected TASK1 system is `DeBERTa-v3-large` on split `dev`, with macro-F1 0.690 (95% bootstrap CI 0.644-0.728).
The best detected TASK2 system is `DeBERTa-v3-base multi-task` on split `dev`, with macro-F1 0.676 (95% bootstrap CI 0.632-0.715).

## Ablation Study
For TASK1, the strongest detected ablation family was `other` with best macro-F1 0.690.
For TASK2, the strongest detected ablation family was `other` with best macro-F1 0.676.

## Error Analysis
The most frequent TASK1 confusion is gold `Clear Reply` predicted as `Ambivalent` (89 examples).
The most frequent TASK2 confusion is gold `Ambivalent` predicted as `Clear Reply` (80 examples).
Qualitative examples are saved under `paper_assets/examples/`.

## Annotator Disagreement
For Task 2, `RoBERTa-large` achieved majority-consensus macro-F1 0.342, unanimous-only macro-F1 0.402, and any-annotator match rate 0.526.

## Limitations
This analysis is limited to saved files available in the repository at notebook execution time. Missing experiments are not reported, and approximate truncation analysis uses whitespace tokenization unless tokenizer-specific lengths are added.