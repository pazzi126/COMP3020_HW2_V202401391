> **VALIDATION ONLY — KHÔNG PHẢI KẾT QUẢ MNIST.** Dữ liệu nhân tạo hoặc load_digits; không dùng các điểm bên dưới để báo cáo bài tập.

# MNIST model selection report

Run type: full. Seed: 202401391. Five shared stratified folds.

## All hyperparameters and fold results

|Representation|Estimator|Parameters|Dimension|Fold accuracies|Mean CV|SD|
|---|---|---|---:|---|---:|---:|
|raw|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|KNN|{"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|KNN|{"n_neighbors": 5, "weights": "distance", "algorithm": "brute"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|KNN|{"n_neighbors": 7, "weights": "distance", "algorithm": "brute"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 4, "max_features": "sqrt"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi1_pooling|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|196|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi1_pooling|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|196|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi1_pooling|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|196|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi2_deskew|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi2_deskew|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|
|phi2_deskew|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|50.0000%, 50.0000%, 50.0000%, 50.0000%, 50.0000%|50.0000%|0.0000%|

## Selection and feature analysis

Warmup winning family: SVC.
Final CV selection: SVC, raw, parameters {'C': 3.0, 'gamma': 'scale', 'kernel': 'rbf'}; dimension 784; mean CV 50.0000%.

Raw features preserve every pixel. Pooling summarizes local ink distribution and reduces dimension from 784 to 196, trading fine strokes for some robustness to local pixel noise. Deskewing retains 784 values while reducing slant variation. Neither transformation learns from labels or from the test set.

phi1_pooling: best mean CV 50.0000%, change +0.0000 percentage points relative to raw, after separate hyperparameter tuning.
phi2_deskew: best mean CV 50.0000%, change +0.0000 percentage points relative to raw, after separate hyperparameter tuning.

Pooling can lose thin distinguishing strokes; deskewing can improve alignment but interpolation and border clipping can also discard information. These deterministic mappings add no new information to the original image; their benefit is an inductive bias and a changed similarity geometry. The observed paired fold differences support a comparison on these splits, not a statistical significance or universal improvement claim.

All design choices and scaling rationale are documented in the source docstring. No learned preprocessing is fitted before splitting. gamma='scale' is computed inside each SVC fit. The winning family and feature representation are chosen using training CV alone. Reported CV is selection-biased; the held-out test provides the final independent performance estimate.

## One final test evaluation

Accuracy: 100.0000% (10000/10000).
96% rubric target met: True.
The model was refitted on all first 10,000 training examples. The final 10,000 examples were predicted exactly once; no baseline or alternative feature test scores were computed.
