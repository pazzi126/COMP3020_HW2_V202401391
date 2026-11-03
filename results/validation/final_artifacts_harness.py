import ast, csv, json, sys, types
from pathlib import Path
import numpy as np
root=Path('/workspace/COMP3020_HW2_V202401391')
source=root/'mnist_assignment.py'
module=types.ModuleType('final_artifact_validation'); module.__file__=str(source)
sys.modules[module.__name__]=module
text=source.read_text(); suffix=text[text.index('# All implementation'):]
exec(compile(suffix,str(source),'exec'),module.__dict__)
# Isolate final-output plumbing: deliberately stub CV scores, not a performance run.
a=np.zeros((2,784),dtype=np.float32); a[1,100:140]=255
module.X=np.broadcast_to(a[:1],(70000,784)); module.y=np.zeros(70000)
module.X_train=np.tile(a,(5000,1)); module.y_train=np.tile(['0','1'],5000)
module.X_test=np.tile(a,(5000,1)); module.y_test=np.tile(['0','1'],5000)
module.evaluate_grid=lambda features, labels, folds, family, representation, jobs, writer, csv_file: [
 {'representation':representation,'family':family,'params':params,'dimension':features.shape[1],
  'fold_accuracies':[.5]*5,'mean_cv_accuracy':.5,'std_cv_accuracy':0.0}
 for params in module.GRIDS[family]]
original=module.make_estimator
calls={'predict':0,'fit':0}
run_dir=root/'results/validation_final_artifacts'
run_dir.mkdir(exist_ok=False)
class CheckedModel:
 def __init__(self): self.inner=original('SVC',module.GRIDS['SVC'][0],1)
 def fit(self,x,y):
  assert len(x)==10000; calls['fit']+=1; self.inner.fit(x,y); return self
 def predict(self,x):
  assert (run_dir/'selection_before_test.json').exists()
  assert len(x)==10000; calls['predict']+=1; return self.inner.predict(x)
module.make_estimator=lambda *args: CheckedModel()
# The production joblib bundle is checked below using an ordinary serializable SVC.
real_dump=module.joblib.dump
def checked_dump(bundle,path,compress):
 bundle['estimator']=bundle['estimator'].inner
 return real_dump(bundle,path,compress=compress)
module.joblib.dump=checked_dump
args=types.SimpleNamespace(smoke=False,jobs=1)
summary=module.experiment(args,run_dir)
assert calls=={'predict':1,'fit':1}
with (run_dir/'test_predictions.csv').open() as f: predictions=list(csv.DictReader(f))
assert len(predictions)==10000
assert predictions[0]['openml_index']=='60000' and predictions[-1]['openml_index']=='69999'
assert np.loadtxt(run_dir/'confusion_matrix.csv',delimiter=',').sum()==10000
bundle=module.joblib.load(run_dir/'final_model.joblib')
assert bundle['representation']=='raw'
assert len(summary['classification_report'])>0
notice={'purpose':'OUTPUT PLUMBING TEST ONLY. Two artificial images, STUBBED CV scores; NOT MNIST RESULTS.',
 'checks':calls,'predictions_saved':len(predictions),'selection_saved_before_predict':True,
 'confusion_matrix_count':10000,'joblib_bundle_loads':True}
(run_dir/'VALIDATION_NOTICE.json').write_text(json.dumps(notice,indent=2))
print(json.dumps(notice,indent=2))
