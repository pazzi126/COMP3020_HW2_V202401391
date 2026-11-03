# MNIST model selection report

Run type: full. Seed: 202401391. Five shared stratified folds.

## All hyperparameters and fold results

|Representation|Estimator|Parameters|Dimension|Fold accuracies|Mean CV|SD|
|---|---|---|---:|---|---:|---:|
|raw|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|97.5000%, 96.4000%, 97.6500%, 95.9500%, 96.4000%|96.7800%|0.7505%|
|raw|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|97.5500%, 96.5500%, 97.7000%, 96.0000%, 96.4000%|96.8400%|0.7462%|
|raw|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|97.5500%, 96.5500%, 97.6500%, 96.0500%, 96.4000%|96.8400%|0.7180%|
|raw|KNN|{"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}|784|95.1500%, 95.5000%, 95.9000%, 93.7500%, 94.1500%|94.8900%|0.9093%|
|raw|KNN|{"n_neighbors": 5, "weights": "distance", "algorithm": "brute"}|784|95.1500%, 95.2000%, 95.9000%, 93.7000%, 94.2500%|94.8400%|0.8656%|
|raw|KNN|{"n_neighbors": 7, "weights": "distance", "algorithm": "brute"}|784|95.2500%, 95.0500%, 95.6000%, 93.2000%, 94.1000%|94.6400%|0.9781%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"}|784|96.1500%, 95.5500%, 95.4500%, 94.1500%, 95.2500%|95.3100%|0.7301%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"}|784|95.8000%, 95.1500%, 95.4500%, 93.8000%, 94.7000%|94.9800%|0.7735%|
|raw|ExtraTrees|{"n_estimators": 200, "min_samples_leaf": 4, "max_features": "sqrt"}|784|95.2500%, 94.2000%, 94.6500%, 93.1500%, 94.3000%|94.3100%|0.7676%|
|phi1_pooling|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|196|97.3000%, 97.0500%, 97.4500%, 96.3500%, 96.3500%|96.9000%|0.5220%|
|phi1_pooling|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|196|97.4500%, 96.8500%, 97.4500%, 96.5500%, 96.5000%|96.9600%|0.4669%|
|phi1_pooling|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|196|97.4500%, 96.9000%, 97.5000%, 96.4500%, 96.6000%|96.9800%|0.4804%|
|phi2_deskew|SVC|{"C": 3.0, "gamma": "scale", "kernel": "rbf"}|784|97.3500%, 97.6000%, 97.6500%, 96.9500%, 97.4500%|97.4000%|0.2784%|
|phi2_deskew|SVC|{"C": 10.0, "gamma": "scale", "kernel": "rbf"}|784|97.5500%, 97.7500%, 97.9500%, 97.1000%, 97.5000%|97.5700%|0.3174%|
|phi2_deskew|SVC|{"C": 30.0, "gamma": "scale", "kernel": "rbf"}|784|97.6000%, 97.8000%, 97.9000%, 97.1500%, 97.4500%|97.5800%|0.2971%|

## Selection and feature analysis

Warmup winning family: SVC.
Final CV selection: SVC, phi2_deskew, parameters {'C': 30.0, 'gamma': 'scale', 'kernel': 'rbf'}; dimension 784; mean CV 97.5800%.

Raw features preserve every pixel. Pooling summarizes local ink distribution and reduces dimension from 784 to 196, trading fine strokes for some robustness to local pixel noise. Deskewing retains 784 values while reducing slant variation. Neither transformation learns from labels or from the test set.

phi1_pooling: best mean CV 96.9800%, change +0.1400 percentage points relative to raw, after separate hyperparameter tuning.
phi2_deskew: best mean CV 97.5800%, change +0.7400 percentage points relative to raw, after separate hyperparameter tuning.

Pooling can lose thin distinguishing strokes; deskewing can improve alignment but interpolation and border clipping can also discard information. These deterministic mappings add no new information to the original image; their benefit is an inductive bias and a changed similarity geometry. The observed paired fold differences support a comparison on these splits, not a statistical significance or universal improvement claim.

All design choices and scaling rationale are documented in the source docstring. No learned preprocessing is fitted before splitting. gamma='scale' is computed inside each SVC fit. The winning family and feature representation are chosen using training CV alone. Reported CV is selection-biased; the held-out test provides the final independent performance estimate.

## One final test evaluation

Accuracy: 97.7300% (9773/10000).
96% rubric target met: True.
The model was refitted on all first 10,000 training examples. The final 10,000 examples were predicted exactly once; no baseline or alternative feature test scores were computed.
