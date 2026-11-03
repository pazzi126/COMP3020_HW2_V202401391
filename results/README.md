# Experiment Results

**The full MNIST experiment completed successfully: 97.73% test accuracy, exceeding the 96% target.**

Official run: `launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/`.

- [English submission report: methods, results, discussion, and all fold scores](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_en.md).
- [Automatically generated report with every fold score](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/report.md).
- [Artifact audit](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/artifact_audit.json): 75 fold records, 15 configurations, five shared folds, 10,000 test predictions, and matching source checksums.
- [Optional Vietnamese discussion](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_vi.md), retained for reference.

The final configuration is an RBF SVC with `C=30`, `gamma="scale"`, and 784-dimensional deskewed features. Mean CV accuracy is 97.58%. The test set was evaluated once, after selection using CV. Subsequent analysis only reads saved results; it does not fit or predict again.

|Location|Purpose|Outcome|
|---|---|---|
|`launches/20261103T115939_full_661/`|Full experiment on MNIST|Completed with exit code 0; test accuracy 97.73%|
|`launches/smoke_20261103T113831Z.log`|Initial MNIST download before the network update|HTTP 403 from api.openml.org; original traceback retained|
|`validation_synthetic/`|CV integration checks using resized and padded load_digits images|75 fits, 15 configurations, five shared folds; no test access; not MNIST results|
|`validation_final_artifacts/`|Checks of refitting and output files using two artificial images and **stubbed CV scores**|One fit and one prediction pass on 10,000 repeated artificial examples; verifies selection ordering and CSV/confusion-matrix/joblib outputs|
|`validation/`|Original assignment, logs, and offline validation harnesses|Offline checks passed; these do not replace the MNIST experiment|

The 100% accuracy in the output-plumbing check comes from **two repeated artificial images**. It only verifies file output behavior and has no relevance to model quality or the assignment's accuracy criterion.

Each experiment creates a separate directory containing its source snapshot, configuration, fold scores, reports, fitted model, and test predictions when applicable. Previous results are not overwritten. See the [main README](../README.md) for installation and launch commands, including how to capture download failures.
