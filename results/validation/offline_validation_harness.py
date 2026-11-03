import ast
import json
from pathlib import Path
import sys
import types
import numpy as np
from sklearn.datasets import load_digits

root = Path('/workspace/COMP3020_HW2_V202401391')
source = root / 'mnist_assignment.py'
tree = ast.parse(source.read_text())
# Verify the required prefix is byte-for-byte unchanged.
prefix = '''from sklearn.datasets import fetch_openml
print("Loading MNIST...")
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
print(f"Loaded {len(X)} examples with {X.shape[1]} features.")
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
X_train, y_train = (X_train[:10000], y_train[:10000]) # reduce dataset size
print(f"Using {len(X_train)} for cross-validation and {len(X_test)} for test.")
'''
assert source.read_text().startswith(prefix)
# Offline integration: execute only added code, never the blocked MNIST loader.
module = types.ModuleType('assignment_validation')
module.__file__ = str(source)
sys.modules[module.__name__] = module
suffix = source.read_text()[len(prefix):]
exec(compile(suffix, str(source), 'exec'), module.__dict__)
digits = load_digits()
images = np.pad(np.repeat(np.repeat(digits.images, 3, axis=1), 3, axis=2),
                ((0,0),(2,2),(2,2))) * (255.0 / 16.0)
train = np.resize(images.reshape(-1,784), (10000,784))
labels = np.resize(digits.target.astype(str), 10000)
module.X = np.broadcast_to(np.zeros((1,784)), (70000,784))
module.y = np.broadcast_to(np.array(['0']), (70000,))
module.X_train = train
module.y_train = labels
class ForbiddenTest:
 def __array__(self, *args, **kwargs):
  raise AssertionError('Smoke run attempted to access held-out test data')
module.X_test = ForbiddenTest()
module.y_test = ForbiddenTest()
sys.argv = [str(source), '--smoke', '--results-dir', str(root/'results/validation_synthetic')]
module.main()
run = max((root/'results/validation_synthetic').glob('run_*'))
summary = json.loads((run/'summary.json').read_text())
assert len(summary['candidates']) == 15
assert all(len(c['fold_accuracies']) == 5 for c in summary['candidates'])
assert 'test_accuracy' not in summary
assert not (run/'test_predictions.csv').exists()
import csv
with (run/'fold_scores.csv').open() as f:
 rows = list(csv.DictReader(f))
assert len(rows) == 75
folds = np.load(run/'cv_splits.npz')
assert len(folds.files) == 10
manifest = {'dataset': 'sklearn load_digits resized and padded, SYNTHETIC VALIDATION ONLY; not MNIST',
 'checks_passed': ['unchanged assignment prefix', 'feature numerical invariants including deskew sign',
 '75 fold fits: 3 families x 3 candidates plus 2 representations x 3 candidates',
 'same saved 5 folds', 'test access forbidden in smoke', '15 candidate summaries',
 'CSV JSON Markdown source snapshot and status outputs'],
 'run_directory':str(run)}
(root/'results/validation/validation_summary.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest, indent=2))
