# Paper Notes

## Repository State

- Repository: `https://github.com/zhaoqyu/Project-MMD2`
- Branch: `mike`
- Local commit checked before writing: `6d6dd20`
- Paper output: `paper/main.tex`

## Source Files Used

Code and configuration:

- `README.md`
- `requirements.txt`
- all files under `configs/*.yaml`
- `src/config.py`
- `src/data.py`
- `src/preprocess.py`
- `src/models.py`
- `src/metrics.py`
- `src/trainer_utils.py`
- `src/evaluate.py`
- `src/utils.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `scripts/ensemble.py`
- `scripts/hierarchical.py`
- `scripts/run_experiments.py`
- `notebooks/analysis_clarity_results.ipynb`

Saved outputs:

- `outputs/runs/*/*/results_summary.json`
- `outputs/runs/*/*/config.yaml`
- `outputs/runs/*/*/split.json`
- `outputs/runs/*/*/label_mapping.json`
- `outputs/runs/*/*/class_weights.json` where present
- saved prediction, metric, report, confusion, probability, logit, and log-history artifacts under `outputs/runs/*/*/artifacts/`

Paper assets:

- `paper_assets/drafts/paper_results_draft.md`
- all CSV files under `paper_assets/tables/`
- all Markdown examples under `paper_assets/examples/`
- all figures under `paper_assets/figures/`

## Tables Used

- `paper_assets/tables/main_results_task1.csv`
- `paper_assets/tables/main_results_task2.csv`
- `paper_assets/tables/compact_paper_results.csv`
- `paper_assets/tables/ablation_results.csv`
- `paper_assets/tables/annotator_disagreement_statistics.csv`
- `paper_assets/tables/confidence_statistics.csv`
- `paper_assets/tables/dataset_statistics.csv`
- `paper_assets/tables/label_distribution_task1.csv`
- `paper_assets/tables/label_distribution_task2.csv`
- `paper_assets/tables/length_statistics.csv`
- `paper_assets/tables/per_class_report_task1.csv`
- `paper_assets/tables/per_class_report_task2.csv`
- `paper_assets/tables/top_confusions_task1.csv`
- `paper_assets/tables/top_confusions_task2.csv`
- `paper_assets/tables/truncation_statistics.csv`

The main paper reports actual split sizes from saved split/prediction artifacts. Some descriptive analysis tables count rows aggregated across multiple saved prediction files, so their repeated counts were not used as raw dataset sizes.

## Figures Included in `main.tex`

- `paper_assets/figures/main_results_task1.pdf`
- `paper_assets/figures/main_results_task2.pdf`
- `paper_assets/figures/task2_majority_vs_unanimous.pdf`
- `paper_assets/figures/per_class_f1_task2.pdf`
- `paper_assets/figures/confusion_matrix_task1_normalized.pdf`
- `paper_assets/figures/confusion_matrix_task2_normalized.pdf`

## Result-Row Checks

- Checked `main_results_task2.csv`, `compact_paper_results.csv`, and `per_class_report_task2.csv`.
- The fixed assets contain zero rows where `task == task2` and a `source_file` points to `artifacts/task1/`.
- `per_class_report_task2.csv` contains the nine fine-grained Task 2 labels: `Explicit`, `Implicit`, `Dodging`, `General`, `Deflection`, `Partial/half-answer`, `Declining to answer`, `Claims ignorance`, and `Clarification`.

## Ambiguities and Cautions

- Task 2 development splits can differ between single-task and multi-task runs because single-task Task 2 splitting stratifies by `evasion_label`, while the multi-task split is stratified by `clarity_label`.
- Bootstrap confidence intervals come from resampling examples in the analysis notebook. They do not measure training-seed variance.
- All saved experiments use seed 42.
- The repository has code for focal loss, seed ensembling, and hierarchical evaluation, but no saved result directories for those methods were present, so they are not reported as completed experiments in `main.tex`.
- The length and truncation analysis uses approximate whitespace tokenization rather than model-tokenizer lengths.

## Claims Needing Human Verification

- Replace `clarity2026` with the official SemEval task paper citation when available.
- Verify the preferred citation for `ailsntua/QEvasion`.
- Verify the authors, arXiv ID, and venue for `"I Never Said That": A dataset, taxonomy and baselines on response clarity classification`.
- Confirm whether the official paper should use `Ambivalent`, `Ambiguous`, or both when describing the middle Task 1 label.
- Confirm whether any saved run corresponds to an official submission before making any leaderboard or submission claim.

## Suggested Next Experiments

- Run the best Task 1 and Task 2 configurations with multiple seeds.
- Run the existing ensemble script after multi-seed experiments.
- Run the existing hierarchical script with the best Task 1 and Task 2 checkpoints.
- Add and run focal-loss configs if time allows.
- Add tokenizer-specific truncation analysis for each model and `max_length`.
- Explore loss weighting or task sampling for the multi-task model instead of fixed `lambda_task2 = 1.0`.

## Validation Summary

- `paper/main.tex` was searched for `TODO`, `FIXME`, `placeholder`, `draft`, `check this`, and `need to verify`; none appeared in the paper body.
- All six figure paths referenced by `\\includegraphics` exist as PDF files in `paper_assets/figures/`.
- All citation keys used in `paper/main.tex` exist in `paper/references.bib`.
- Every table and figure label in `paper/main.tex` is referenced in the text.
- Key reported metrics were checked against `main_results_task1.csv`, `main_results_task2.csv`, `annotator_disagreement_statistics.csv`, and the top-confusion/per-class tables.
- LaTeX compilation succeeded with TeX Live via `latexmk`; the compiled PDF is `paper/build/main.pdf`.
