from sklearn.datasets import fetch_openml
print("Loading MNIST...")
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
print(f"Loaded {len(X)} examples with {X.shape[1]} features.")
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
X_train, y_train = (X_train[:10000], y_train[:10000]) # reduce dataset size
print(f"Using {len(X_train)} for cross-validation and {len(X_test)} for test.")

# All implementation is added AFTER the assignment's unchanged loading code.
"""COMP3020 HW2, Section 6. Single-file submission.

Run: python -u mnist_assignment.py
Quick installation: python -m pip install numpy scipy scikit-learn pandas
Validated versions are recorded in each run's metadata, rather than silently
depending on a machine's default versions. The OpenML download is cached by sklearn.

Decisions made BEFORE looking at test accuracy:
* Five stratified shuffled folds, seed 202401391; exactly the same index pairs
  for every candidate and every representation. No GridSearchCV or equivalents.
* Scale intensities by the known constant 255 (not a learned statistic).
  This preserves the physical meaning of pixels and introduces no fold leakage.
* RBF SVC: C in {3, 10, 30}, gamma='scale'. A nonlinear model is appropriate
  for MNIST; gamma is computed by sklearn from each fold's training data only.
* Distance-weighted Euclidean KNN: k in {3, 5, 7}. This gives a strong local
  similarity baseline; brute force avoids unreliable high-dimensional trees.
* ExtraTrees: min_samples_leaf in {1, 2, 4}, 200 trees, sqrt feature sampling.
  This is a nonlinear ensemble baseline with no distance scaling assumption.
* phi1: nonoverlapping 2x2 average pooling, 196 features. It captures local ink
  distribution, suppresses small pixel fluctuations, but may erase fine strokes.
* phi2: moment-based horizontal deskewing, 784 features. It removes variation
  in slant, potentially making within-class distances more consistent. This is
  implemented below with NumPy moments and bilinear interpolation, not a feature
  extraction library. Each image is transformed independently, without labels.
* Compare raw/phi1/phi2 using the winning warmup estimator family; retune the
  same three hyperparameters for each. Raw scores are reused, not rerun.
* Highest mean CV accuracy wins; exact ties use earlier declared candidates
  and representations. No test accuracy is used for tuning or feature design.
* The warmup asks for a test score, but the later instruction explicitly says
  to evaluate the final selected model on test only once. Accordingly, only
  the final CV-selected representation/model receives a test evaluation.
* The >=96% threshold is a goal, never a reason to retune using test results.

Every experiment creates a unique results/run_* directory. CSV rows are flushed
after each fold and status.json records failures. --smoke uses only 300 training
images and NEVER evaluates test; it is not an assignment result. The full run
always uses the exact prescribed 10,000 train / final 10,000 test examples.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEED = 202401391
GRIDS = {
    "SVC": [{"C": c, "gamma": "scale", "kernel": "rbf"} for c in (3.0, 10.0, 30.0)],
    "KNN": [{"n_neighbors": k, "weights": "distance", "algorithm": "brute"}
            for k in (3, 5, 7)],
    "ExtraTrees": [{"n_estimators": 200, "min_samples_leaf": leaf,
                    "max_features": "sqrt"} for leaf in (1, 2, 4)],
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def normalized_pixels(values):
    return np.ascontiguousarray(values, dtype=np.float32) / np.float32(255)


def pooled_features(pixels):
    """phi1: 14x14 spatial grid of mean intensities of disjoint 2x2 cells."""
    images = pixels.reshape(-1, 28, 28)
    return np.ascontiguousarray(images.reshape(-1, 14, 2, 14, 2)
                                .mean(axis=(2, 4)).reshape(-1, 196))


def deskew_features(pixels):
    """phi2: remove horizontal slant with image moments and inverse sampling.

    mu11 / mu02 estimates the slope of column position against row position.
    Output at (r,c) samples input at (r,c + slope*(r-13.5)). Row coordinates
    do not change, so bilinear interpolation reduces to linear interpolation
    between adjacent columns. Out-of-image neighbors contribute zero.
    Blank/degenerate images use zero shear. Nothing is fitted across images.
    """
    images = pixels.reshape(-1, 28, 28)
    output = np.empty_like(images)
    rows, cols = np.mgrid[:28, :28].astype(np.float32)
    for start in range(0, len(images), 512):
        batch = images[start:start + 512]
        mass = batch.sum(axis=(1, 2))
        safe_mass = np.where(mass > 1e-8, mass, 1)
        row_mean = (batch * rows).sum(axis=(1, 2)) / safe_mass
        col_mean = (batch * cols).sum(axis=(1, 2)) / safe_mass
        dr = rows[None] - row_mean[:, None, None]
        dc = cols[None] - col_mean[:, None, None]
        mu11 = (batch * dr * dc).sum(axis=(1, 2))
        mu02 = (batch * dr * dr).sum(axis=(1, 2))
        slope = np.divide(mu11, mu02, out=np.zeros_like(mu11), where=mu02 > 1e-8)
        source_col = cols[None] + slope[:, None, None] * (rows[None] - 13.5)
        left = np.floor(source_col).astype(np.int32)
        fraction = source_col - left
        a = np.take_along_axis(batch, np.clip(left, 0, 27), axis=2)
        b = np.take_along_axis(batch, np.clip(left + 1, 0, 27), axis=2)
        a = a * ((left >= 0) & (left < 28))
        b = b * ((left + 1 >= 0) & (left + 1 < 28))
        output[start:start + len(batch)] = (1 - fraction) * a + fraction * b
    return np.ascontiguousarray(output.reshape(-1, 784))


TRANSFORMS = {"raw": lambda pixels: pixels,
              "phi1_pooling": pooled_features,
              "phi2_deskew": deskew_features}


def self_checks():
    """Numerical checks independent of MNIST labels and test outcomes."""
    zero = np.zeros((2, 784), dtype=np.float32)
    assert pooled_features(zero).shape == (2, 196)
    np.testing.assert_array_equal(deskew_features(zero), zero)
    np.testing.assert_array_equal(pooled_features(np.ones_like(zero)),
                                  np.ones((2, 196), dtype=np.float32))
    ramp = np.arange(784, dtype=np.float32).reshape(1, 784)
    assert pooled_features(ramp)[0, 0] == np.mean([0, 1, 28, 29])
    upright = np.zeros((1, 28, 28), dtype=np.float32)
    upright[:, 4:24, 13:15] = 1
    np.testing.assert_allclose(deskew_features(upright.reshape(1, 784)),
                               upright.reshape(1, 784), atol=1e-6)
    # A synthetic slanted stroke should have lower row/column covariance after
    # correction. This also checks the sign of the inverse shear mapping.
    slanted = np.zeros((1, 28, 28), dtype=np.float32)
    for r in range(4, 24):
        slanted[0, r, 8 + (r - 4) // 2] = 1
    fixed = deskew_features(slanted.reshape(1, 784)).reshape(28, 28)
    def covariance(im):
        rr, cc = np.mgrid[:28, :28]
        mass = im.sum()
        return float((im * (rr - (im * rr).sum()/mass)
                      * (cc - (im * cc).sum()/mass)).sum()/mass)
    assert abs(covariance(fixed)) < abs(covariance(slanted[0])) * 0.1
    assert np.isfinite(fixed).all() and fixed.min() >= 0 and fixed.max() <= 1
    print("Numerical feature checks passed.")


def make_estimator(family, params, jobs):
    if family == "SVC":
        return SVC(**params, cache_size=512)
    if family == "KNN":
        return KNeighborsClassifier(**params, n_jobs=jobs)
    if family == "ExtraTrees":
        return ExtraTreesClassifier(**params, random_state=SEED, n_jobs=jobs)
    raise ValueError(family)


def evaluate_grid(features, labels, folds, family, representation, jobs, writer, csv_file):
    candidates = []
    for index, params in enumerate(GRIDS[family]):
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(folds, 1):
            # Fresh estimator for EVERY fold; no warm start or shared fitted state.
            estimator = make_estimator(family, params, jobs)
            started = time.perf_counter()
            estimator.fit(features[train_idx], labels[train_idx])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = estimator.predict(features[valid_idx])
            predict_seconds = time.perf_counter() - started
            score = float(accuracy_score(labels[valid_idx], prediction))
            scores.append(score)
            writer.writerow({"representation": representation, "family": family,
                             "candidate": index, "params": json.dumps(params, sort_keys=True),
                             "fold": fold, "n_train": len(train_idx), "n_valid": len(valid_idx),
                             "accuracy": score, "fit_seconds": fit_seconds,
                             "predict_seconds": predict_seconds})
            csv_file.flush()
            print(f"{representation:14s} {family:10s} {params} fold={fold} "
                  f"accuracy={score:.6f} fit={fit_seconds:.2f}s predict={predict_seconds:.2f}s")
        result = {"representation": representation, "family": family,
                  "params": params, "dimension": int(features.shape[1]),
                  "fold_accuracies": scores, "mean_cv_accuracy": float(np.mean(scores)),
                  "std_cv_accuracy": float(np.std(scores, ddof=1))}
        candidates.append(result)
        print(f"  Mean CV={result['mean_cv_accuracy']:.6f}; "
              f"sample SD={result['std_cv_accuracy']:.6f}")
    return candidates


def report_text(summary):
    lines = ["# MNIST model selection report", "",
             f"Run type: {summary['mode']}. Seed: {SEED}. Five shared stratified folds.",
             "", "## All hyperparameters and fold results", "",
             "|Representation|Estimator|Parameters|Dimension|Fold accuracies|Mean CV|SD|",
             "|---|---|---|---:|---|---:|---:|"]
    for c in summary["candidates"]:
        scores = ", ".join(f"{s:.4%}" for s in c["fold_accuracies"])
        lines.append(f"|{c['representation']}|{c['family']}|{json.dumps(c['params'])}|"
                     f"{c['dimension']}|{scores}|{c['mean_cv_accuracy']:.4%}|"
                     f"{c['std_cv_accuracy']:.4%}|")
    winner = summary["selected"]
    lines += ["", "## Selection and feature analysis", "",
              f"Warmup winning family: {summary['warmup_best']['family']}.",
              f"Final CV selection: {winner['family']}, {winner['representation']}, "
              f"parameters {winner['params']}; dimension {winner['dimension']}; "
              f"mean CV {winner['mean_cv_accuracy']:.4%}.", "",
              "Raw features preserve every pixel. Pooling summarizes local ink distribution "
              "and reduces dimension from 784 to 196, trading fine strokes for some robustness "
              "to local pixel noise. Deskewing retains 784 values while reducing slant variation. "
              "Neither transformation learns from labels or from the test set.", ""]
    raw = summary["representation_best"]["raw"]["mean_cv_accuracy"]
    for name in ("phi1_pooling", "phi2_deskew"):
        score = summary["representation_best"][name]["mean_cv_accuracy"]
        delta = 100 * (score - raw)
        lines.append(f"{name}: best mean CV {score:.4%}, change {delta:+.4f} percentage "
                     "points relative to raw, after separate hyperparameter tuning.")
    lines += ["", "Pooling can lose thin distinguishing strokes; deskewing can improve "
              "alignment but interpolation and border clipping can also discard information. "
              "These deterministic mappings add no new information to the original image; "
              "their benefit is an inductive bias and a changed similarity geometry. "
              "The observed paired fold differences support a comparison on these splits, "
              "not a statistical significance or universal improvement claim.", "",
              "All design choices and scaling rationale are documented in the source docstring. "
              "No learned preprocessing is fitted before splitting. gamma='scale' is computed "
              "inside each SVC fit. The winning family and feature representation are chosen "
              "using training CV alone. Reported CV is selection-biased; the held-out test "
              "provides the final independent performance estimate.", ""]
    if "test_accuracy" in summary:
        lines += ["## One final test evaluation", "",
                  f"Accuracy: {summary['test_accuracy']:.4%} "
                  f"({summary['test_correct']}/10000).",
                  f"96% rubric target met: {summary['meets_96_percent_target']}.",
                  "The model was refitted on all first 10,000 training examples. "
                  "The final 10,000 examples were predicted exactly once; no baseline "
                  "or alternative feature test scores were computed.", ""]
    else:
        lines += ["Smoke validation only: no test evaluation, no assignment performance claim.", ""]
    return "\n".join(lines)


def experiment(args, run_dir):
    self_checks()
    if X.shape != (70000, 784) or len(y) != 70000:
        raise ValueError("Expected canonical OpenML MNIST ordering and shape (70000,784).")
    pixels = normalized_pixels(X_train)
    labels = np.asarray(y_train).astype(str)
    if args.smoke:
        # Deterministic per-class sampling from TRAINING ONLY, retaining all 10 classes.
        rng = np.random.default_rng(SEED)
        selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), 30,
                                                      replace=False) for c in np.unique(labels)]))
        pixels, labels = pixels[selected], labels[selected]
        np.save(run_dir / "smoke_training_indices.npy", selected)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(pixels, labels))
    coverage = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(np.sort(coverage), np.arange(len(labels)))
    for train_idx, valid_idx in folds:
        assert not np.intersect1d(train_idx, valid_idx).size
    np.savez_compressed(run_dir / "cv_splits.npz",
                        **{f"{part}_{i+1}": indices for i, pair in enumerate(folds)
                           for part, indices in zip(("train", "validation"), pair)})
    metadata = {"seed": SEED, "mode": "smoke" if args.smoke else "full",
                "training_examples": len(labels), "test_examples": 10000,
                "folds": 5, "jobs": args.jobs, "hyperparameters": GRIDS,
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "training_labels_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest()}
    write_json(run_dir / "metadata.json", metadata)
    fields = ["representation", "family", "candidate", "params", "fold", "n_train",
              "n_valid", "accuracy", "fit_seconds", "predict_seconds"]
    candidates = []
    with (run_dir / "fold_scores.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for family in GRIDS:
            candidates.extend(evaluate_grid(pixels, labels, folds, family, "raw",
                                            args.jobs, writer, csv_file))
            write_json(run_dir / "cv_candidates.json", candidates)
        warmup_best = max(candidates, key=lambda c: c["mean_cv_accuracy"])
        family = warmup_best["family"]
        representation_best = {"raw": warmup_best}
        print(f"Warmup winner: {family}, {warmup_best['params']}")
        for name in ("phi1_pooling", "phi2_deskew"):
            transformed = TRANSFORMS[name](pixels)
            assert np.isfinite(transformed).all()
            results = evaluate_grid(transformed, labels, folds, family, name,
                                    args.jobs, writer, csv_file)
            candidates.extend(results)
            representation_best[name] = max(results, key=lambda c: c["mean_cv_accuracy"])
            write_json(run_dir / "cv_candidates.json", candidates)
    selected = max(representation_best.values(), key=lambda c: c["mean_cv_accuracy"])
    summary = {"mode": metadata["mode"], "warmup_best": warmup_best,
               "representation_best": representation_best, "selected": selected,
               "candidates": candidates}
    # Persist the selection BEFORE any test operation, making the selection auditable.
    write_json(run_dir / "selection_before_test.json", summary)
    if not args.smoke:
        print(f"Refitting final selection: {selected}")
        final_model = make_estimator(selected["family"], selected["params"], args.jobs)
        started = time.perf_counter()
        final_model.fit(TRANSFORMS[selected["representation"]](pixels), labels)
        summary["refit_seconds"] = time.perf_counter() - started
        # Only this final selected model is ever applied to the held-out test set.
        test_features = TRANSFORMS[selected["representation"]](normalized_pixels(X_test))
        started = time.perf_counter()
        prediction = final_model.predict(test_features)
        summary["test_predict_seconds"] = time.perf_counter() - started
        truth = np.asarray(y_test).astype(str)
        summary["test_accuracy"] = float(accuracy_score(truth, prediction))
        summary["test_correct"] = int(np.sum(truth == prediction))
        summary["meets_96_percent_target"] = summary["test_accuracy"] >= .96
        summary["classification_report"] = classification_report(
            truth, prediction, labels=[str(i) for i in range(10)],
            output_dict=True, zero_division=0)
        matrix = confusion_matrix(truth, prediction, labels=[str(i) for i in range(10)])
        np.savetxt(run_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
        with (run_dir / "test_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["openml_index", "true_label", "predicted_label", "correct"])
            writer.writerows((60000 + i, a, b, int(a == b))
                             for i, (a, b) in enumerate(zip(truth, prediction)))
        # Bundle records the custom preprocessing needed before calling predict.
        joblib.dump({"estimator": final_model, "representation": selected["representation"],
                     "input_scaling": "float32 / 255, then TRANSFORMS[representation]",
                     "seed": SEED}, run_dir / "final_model.joblib", compress=3)
        print(f"FINAL TEST ACCURACY: {summary['test_accuracy']:.4%}")
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(report_text(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="300 training samples; no test evaluation")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.results_dir / f"run_{stamp}_{'smoke' if args.smoke else 'full'}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # Snapshot the exact submitted source alongside results for reproducibility.
    (run_dir / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Results directory: {run_dir}")
            write_json(run_dir / "status.json", {"status": "running", "started_utc": stamp,
                                                 "argv": sys.argv})
            try:
                with threadpool_limits(limits=args.jobs):
                    experiment(args, run_dir)
            except BaseException as exc:
                traceback.print_exc()
                write_json(run_dir / "status.json", {"status": "failed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__, "error": str(exc)})
                raise
            write_json(run_dir / "status.json", {"status": "completed",
                "elapsed_seconds": time.perf_counter() - started, "started_utc": stamp})
            print(f"Completed in {time.perf_counter() - started:.1f}s; saved to {run_dir}")


if __name__ == "__main__":
    main()
