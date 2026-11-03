> **VALIDATION ONLY — KHÔNG PHẢI KẾT QUẢ MNIST.** Dữ liệu nhân tạo hoặc load_digits; không dùng các điểm bên dưới để báo cáo bài tập.

# MNIST model selection report

Run type: smoke. Seed: 202401391. Five shared stratified folds.

## All hyperparameters and fold results

|Representation|Estimator|Parameters|Dimension|Fold accuracies|Mean CV|SD|
|---|---|---|---:|---|---:|---:|
|raw|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 95.0000%|97.0000%|2.7386%|
|raw|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 95.0000%|97.0000%|2.7386%|
|raw|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 95.0000%|97.0000%|2.7386%|
|raw|KNN|{"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}|784|100.0000%, 98.3333%, 93.3333%, 93.3333%, 96.6667%|96.3333%|2.9814%|
|raw|KNN|{"n_neighbors": 5, "weights": "distance", "algorithm": "brute"}|784|98.3333%, 100.0000%, 93.3333%, 93.3333%, 96.6667%|96.3333%|2.9814%|
|raw|KNN|{"n_neighbors": 7, "weights": "distance", "algorithm": "brute"}|784|96.6667%, 100.0000%, 88.3333%, 93.3333%, 95.0000%|94.6667%|4.3141%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"}|784|96.6667%, 100.0000%, 90.0000%, 95.0000%, 93.3333%|95.0000%|3.7268%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"}|784|98.3333%, 100.0000%, 90.0000%, 96.6667%, 91.6667%|95.3333%|4.3141%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 4, "max_features": "sqrt"}|784|95.0000%, 98.3333%, 90.0000%, 95.0000%, 93.3333%|94.3333%|3.0277%|
|phi1_pooling|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|196|98.3333%, 100.0000%, 91.6667%, 98.3333%, 96.6667%|97.0000%|3.2059%|
|phi1_pooling|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|196|98.3333%, 100.0000%, 93.3333%, 98.3333%, 95.0000%|97.0000%|2.7386%|
|phi1_pooling|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|196|98.3333%, 100.0000%, 93.3333%, 98.3333%, 95.0000%|97.0000%|2.7386%|
|phi2_deskew|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 96.6667%|97.3333%|2.5276%|
|phi2_deskew|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 96.6667%|97.3333%|2.5276%|
|phi2_deskew|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|98.3333%, 100.0000%, 93.3333%, 98.3333%, 96.6667%|97.3333%|2.5276%|

## Selection and feature analysis

Warmup winning family: SVC.
Final CV selection: SVC, phi2_deskew, parameters {'C': 3.0, 'gamma': 'scale', 'kernel': 'rbf'}; dimension 784; mean CV 97.3333%.

Raw features preserve every pixel. Pooling summarizes local ink distribution and reduces dimension from 784 to 196, trading fine strokes for some robustness to local pixel noise. Deskewing retains 784 values while reducing slant variation. Neither transformation learns from labels or from the test set.

phi1_pooling: best mean CV 97.0000%, change +0.0000 percentage points relative to raw, after separate hyperparameter tuning.
phi2_deskew: best mean CV 97.3333%, change +0.3333 percentage points relative to raw, after separate hyperparameter tuning.

Pooling can lose thin distinguishing strokes; deskewing can improve alignment but interpolation and border clipping can also discard information. These deterministic mappings add no new information to the original image; their benefit is an inductive bias and a changed similarity geometry. The observed paired fold differences support a comparison on these splits, not a statistical significance or universal improvement claim.

All design choices and scaling rationale are documented in the source docstring. No learned preprocessing is fitted before splitting. gamma='scale' is computed inside each SVC fit. The winning family and feature representation are chosen using training CV alone. Reported CV is selection-biased; the held-out test provides the final independent performance estimate.

Smoke validation only: no test evaluation, no assignment performance claim.
