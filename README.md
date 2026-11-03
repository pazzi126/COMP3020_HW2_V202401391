# COMP3020 HW2 — MNIST Classification

This repository implements Section 6 of the assignment: manual cross-validation, model selection, and hand-crafted feature engineering for MNIST.

**Final test accuracy: 97.73% (9,773/10,000), exceeding the 96% target.** The selected model is an RBF SVC with `C=30`, `gamma="scale"`, and 784-dimensional deskewed features. Its mean five-fold cross-validation accuracy is **97.58%**.

## Code and report

- **Single-file implementation:** [mnist_assignment.py](mnist_assignment.py). The assignment's original loading and splitting code is preserved at the beginning; all implementation follows it.
- **English assignment report:** [discussion_en.md](results/launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_en.md), including methods, design decisions, feature definitions, analysis, final test results, and all 75 fold scores.
- **Results guide:** [results/README.md](results/README.md).
- **Original assignment:** [assignment_original.txt](results/validation/assignment_original.txt).

Only `mnist_assignment.py` is the implementation file for submission. Files under `results/` preserve experimental evidence and supporting analysis.

## Installation and execution

The experiment was run with Python 3.12 and these dependency versions:

```bash
python3 -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 pandas==2.2.3 joblib==1.5.3 threadpoolctl==3.6.0
```

Run the following Bash commands from the repository root. OpenML requires network access on the first run; scikit-learn caches the downloaded dataset. This launcher also records download errors that occur before the appended implementation executes.

```bash
mkdir -p results/launches
launch="results/launches/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$launch"
set -o pipefail
python3 -u mnist_assignment.py --jobs 4 --results-dir "$launch" 2>&1 | tee "$launch/console.log"
status=${PIPESTATUS[0]}
echo "$status" > "$launch/exit_code.txt"
```

For a quick workflow check, add `--smoke` to the Python command. Smoke mode uses 300 training images, performs all 75 CV fits, and **does not evaluate the test set**. Smoke scores are not assignment results.

Each run creates a unique directory without overwriting previous results. Directory timestamps use UTC. No background service or additional Git worktree is required.

## Assignment requirements

|Requirement|Implementation|
|---|---|
|First 10,000 training examples and final 10,000 test examples|Preserves the supplied split; verifies the full dataset shape of 70,000 × 784|
|Manual 5- or 10-fold CV|Five-fold `StratifiedKFold` with explicit loops and a fresh estimator per fold; no automatic search classes|
|At least three estimators, each with three configurations|RBF SVC: C=3/10/30; KNN: k=3/5/7; ExtraTrees: min_samples_leaf=1/2/4|
|Every fold accuracy and mean for every configuration|Saved in `fold_scores.csv`, `cv_candidates.json`, and the reports|
|Two manually implemented feature transformations|NumPy 2×2 average pooling → 196 features; moment-based deskewing and interpolation → 784 features|
|Same folds for representation comparisons|Reuses identical index pairs, saved in `cv_splits.npz`|
|Use the best warmup estimator|Selects the family using raw-pixel mean CV accuracy, then retunes three configurations per representation|
|No test leakage|Image-local transformations, constant scaling by 1/255, and SVC gamma computed within each training fold|
|Refit on all training examples and test once|Saves the CV selection before transforming or predicting the test set; one final test prediction pass|
|Reproducibility|Seed 202401391, dependency versions, training/source checksums, source snapshot, and fold indices|
|Explain decisions and observations|Detailed source comments and English report based on saved results|
|At least 96% test accuracy|Achieved 97.73%; no retuning based on the test score|

The warmup requests a test evaluation, while the later instructions require evaluating the final selected model on the test set **only once**. The implementation follows the latter restriction: all warmup and feature comparisons use CV, and only the final selected configuration receives a test evaluation.

There are **75 CV fits**: 45 for warmup (3 × 3 × 5) and 30 for feature engineering (2 × 3 × 5). Raw scores for the winning family are reused. A full run adds one fit on all 10,000 training examples and one test prediction pass. SVC has a 512 MB cache; supported libraries use at most four threads by default.

## Results

|Representation|Features|Best SVC C|Mean CV accuracy|
|---|---:|---:|---:|
|Raw pixels|784|10|96.84%|
|2×2 average pooling|196|30|96.98%|
|Moment-based deskewing|784|30|**97.58%**|

The full MNIST run started on November 3, 2026, at 19:00 ICT (UTC+7) and completed in approximately 188 seconds after loading the data. The final configuration achieved **97.73% test accuracy**. All 75 fold records and saved artifacts were checked for consistency, and the implementation matches the run's source snapshot.

The official run is stored under:

```text
results/launches/20261103T115939_full_661/
└── run_20261103T120001_917778Z_full_667/
```

Its path is also recorded in [results/latest_mnist_run.txt](results/latest_mnist_run.txt). CV comparison and confusion-matrix figures are available as PNG and SVG files. Further analysis reads saved outputs without refitting models or predicting test examples again.

## Saved artifacts

|File|Contents|
|---|---|
|`metadata.json`|Hyperparameter grids, seed, versions, checksums, sample counts, and thread count|
|`cv_splits.npz`|Training and validation indices for the five shared folds|
|`fold_scores.csv`|Accuracy and timing for every fold, flushed immediately after each fold|
|`cv_candidates.json`|Completed candidate results, preserving progress if a later step fails|
|`selection_before_test.json`|Configuration selected using CV before test evaluation|
|`summary.json`, `report.md`|CV scores, selected configuration, feature analysis, and final test results|
|`discussion_en.md`|Detailed English submission report and all fold scores|
|`test_predictions.csv`|OpenML indices 60000–69999, true labels, predictions, and correctness|
|`confusion_matrix.csv`|Rows are true labels; columns are predictions; digit order is 0–9|
|`final_model.joblib`|Refitted estimator and representation name; reuse requires the source's scaling and feature transformation|
|`source_snapshot.py`, `run.log`, `status.json`|Exact run source, execution log, completion status, and elapsed time|
|`artifact_audit.json`|Consistency checks on saved scores, splits, predictions, and source checksum|

Feature comparisons use **cross-validation**, not test scores. Fold standard deviations are not confidence intervals, and selection using CV can make the winning CV score optimistic. Only load trusted joblib files.

## Historical validation runs

The initial MNIST download failed with HTTP 403 before network access was updated. Its traceback remains in `results/launches/smoke_20261103T113831Z.log`.

Earlier offline integration checks used resized and padded `sklearn.datasets.load_digits` images and are stored in `results/validation_synthetic/`. These are **not MNIST results**. Separate output-plumbing checks used two artificial images and stubbed CV scores in `results/validation_final_artifacts/`; their accuracy is not an assignment performance result. The assignment report uses only the completed full MNIST run above.
