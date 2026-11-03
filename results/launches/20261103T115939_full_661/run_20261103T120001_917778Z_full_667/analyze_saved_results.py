"""Audit and visualize saved results only; NEVER refit or predict any model."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/comp3020-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/comp3020-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path(__file__).resolve().parent
root = next(parent for parent in run.parents if (parent/'mnist_assignment.py').is_file())
summary = json.loads((run/'summary.json').read_text())
metadata = json.loads((run/'metadata.json').read_text())
status = json.loads((run/'status.json').read_text())
selection = json.loads((run/'selection_before_test.json').read_text())
assert status['status'] == 'completed'
assert metadata['mode']=='full' and metadata['training_examples']==10000
assert metadata['test_examples']==10000
assert 'test_accuracy' not in selection
assert selection['selected'] == summary['selected']
assert hashlib.sha256((run/'source_snapshot.py').read_bytes()).hexdigest()==metadata['source_sha256']
assert (run/'source_snapshot.py').read_bytes()==(root/'mnist_assignment.py').read_bytes()
with (run/'fold_scores.csv').open() as f: rows=list(csv.DictReader(f))
assert len(rows)==75 and len(summary['candidates'])==15
folds=np.load(run/'cv_splits.npz')
valids=[]
for i in range(1,6):
 train,valid=folds[f'train_{i}'],folds[f'validation_{i}']
 assert len(train)==8000 and len(valid)==2000
 assert not np.intersect1d(train,valid).size
 assert np.all((train>=0)&(train<10000)) and np.all((valid>=0)&(valid<10000))
 valids.extend(valid)
np.testing.assert_array_equal(np.sort(valids),np.arange(10000))
for candidate in summary['candidates']:
 actual=[r for r in rows if r['representation']==candidate['representation']
         and r['family']==candidate['family'] and json.loads(r['params'])==candidate['params']]
 assert len(actual)==5
 np.testing.assert_allclose([float(r['accuracy']) for r in actual],candidate['fold_accuracies'])
 assert np.isclose(np.mean(candidate['fold_accuracies']),candidate['mean_cv_accuracy'])
with (run/'test_predictions.csv').open() as f: predictions=list(csv.DictReader(f))
assert len(predictions)==10000
assert [int(r['openml_index']) for r in predictions]==list(range(60000,70000))
correct=sum(r['true_label']==r['predicted_label'] for r in predictions)
assert correct==summary['test_correct']==9773
assert np.isclose(correct/10000,summary['test_accuracy'])
matrix=np.loadtxt(run/'confusion_matrix.csv',delimiter=',',dtype=int)
expected=np.zeros((10,10),dtype=int)
for row in predictions:
 expected[int(row['true_label']),int(row['predicted_label'])]+=1
np.testing.assert_array_equal(matrix,expected)
assert matrix.trace()==correct
assert (run/'run.log').read_text().count('FINAL TEST ACCURACY:')==1
assert (run.parent/'exit_code.txt').read_text().strip()=='0'
reps=summary['representation_best']
fig,ax=plt.subplots(figsize=(7.8,4.5),layout='constrained')
names=['Raw pixels','2x2 pooling','Deskew']
values=[c['mean_cv_accuracy']*100 for c in reps.values()]
errors=[c['std_cv_accuracy']*100 for c in reps.values()]
ax.bar(names,values,yerr=errors,capsize=6,color=['#6c809a','#e9a23b','#329b80'])
for i,v in enumerate(values): ax.text(i,v+errors[i]+.12,f'{v:.2f}%',ha='center')
ax.set(ylim=(94,100),ylabel='Validation accuracy (%)',
       title='MNIST: best SVC per representation, same 5 folds')
ax.text(.02,.02,'Error bars: sample SD across folds (not confidence intervals).',
        transform=ax.transAxes,fontsize=8)
ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
for extension in ['png','svg']: fig.savefig(run/f'cv_comparison.{extension}',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
image=ax.imshow(matrix,cmap='Blues')
for i in range(10):
 for j in range(10):
  ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,
          color='white' if matrix[i,j]>matrix.max()*.5 else 'black')
ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',
       title=f'Final MNIST test confusion matrix — {summary["test_accuracy"]:.2%}')
fig.colorbar(image,ax=ax,label='Number of examples')
for extension in ['png','svg']: fig.savefig(run/f'confusion_matrix.{extension}',dpi=180)
plt.close(fig)
confusions=sorted([(int(matrix[i,j]),i,j) for i in range(10) for j in range(10) if i!=j],reverse=True)[:8]
lines=['# Thảo luận kết quả MNIST thật','',
 f'Full run: `{run.name}`. Chương trình hoàn tất trong {status["elapsed_seconds"]:.1f} giây sau bước tải dữ liệu.',
 'Dữ liệu: OpenML mnist_784 version=1, 70.000 ảnh; dùng đúng 10.000 ảnh train đầu và 10.000 ảnh test cuối. Không dùng load_digits hay dữ liệu nhân tạo trong run này.','',
 '## Lựa chọn mô hình bằng CV','',
 '|Family trên raw pixel|Siêu tham số tốt nhất|CV trung bình|',
 '|---|---|---:|']
for family in ['SVC','KNN','ExtraTrees']:
 c=max([c for c in summary['candidates'] if c['representation']=='raw' and c['family']==family],key=lambda c:c['mean_cv_accuracy'])
 lines.append(f'|{family}|`{json.dumps(c["params"])}`|{c["mean_cv_accuracy"]:.2%}|')
lines+=['','SVC thắng warmup. Theo yêu cầu đề, phần feature engineering chỉ giữ family SVC và tune lại C = 3, 10, 30 cho mỗi biểu diễn. Trường hợp mean bằng nhau chọn cấu hình xuất hiện trước; không dùng test để phá hòa.','',
 '## So sánh biểu diễn trên cùng 5 fold','',
 '|Biểu diễn|Chiều|C tốt nhất|Mean CV|Chênh lệch so với raw (điểm phần trăm)|',
 '|---|---:|---:|---:|---:|']
raw=reps['raw']['mean_cv_accuracy']
for name,c in reps.items():
 lines.append(f'|{name}|{c["dimension"]}|{c["params"]["C"]:g}|{c["mean_cv_accuracy"]:.2%}|{100*(c["mean_cv_accuracy"]-raw):+.2f}|')
lines+=['','Pooling chia ảnh thành lưới 14×14, lấy trung bình từng ô 2×2. Phép biến đổi giữ phân bố mực theo vị trí và làm mượt nhiễu pixel, giảm chiều từ 784 xuống 196; đổi lại có thể mất nét mảnh. CV tăng nhẹ 0,14 điểm phần trăm, chưa đủ để kết luận cải thiện có ý nghĩa thống kê.','',
 'Deskew ước lượng độ nghiêng bằng moment từng ảnh rồi shear theo phương ngang, nội suy bằng NumPy. Nó giảm khác biệt hình học do cách viết nghiêng. CV tăng 0,74 điểm phần trăm so với raw, và điểm của cấu hình deskew thắng cao hơn cấu hình raw thắng trên cả 5 fold. Đây là bằng chứng trên các fold đã chọn, không phải khẳng định mọi tập dữ liệu đều cải thiện.','',
 'Các phép biến đổi là hàm xác định của ảnh gốc, không bổ sung thông tin mới. Lợi ích đến từ cách biểu diễn làm thay đổi khoảng cách/inductive bias của mô hình. Deskew vẫn có thể làm mất thông tin do nội suy và cắt biên.','',
 '## Mô hình cuối và test','',
 f'- SVC RBF, `C=30`, `gamma="scale"`, deskew 784 chiều.',
 f'- Mean CV: {summary["selected"]["mean_cv_accuracy"]:.2%}.',
 f'- Refit trên đủ 10.000 ảnh train, sau đó một lượt predict trên 10.000 ảnh test.',
 f'- Test accuracy: **{summary["test_accuracy"]:.2%} ({correct}/10000)**; {10000-correct} ảnh sai.',
 '- Vượt ngưỡng 96% của rubric. Không thay đổi siêu tham số hoặc feature sau khi xem test.',
 '- Điểm test cao hơn CV không có nghĩa là leakage: refit dùng 10.000 mẫu, trong khi mỗi fold dùng 8.000 mẫu; thành phần dữ liệu và độ khó cũng khác nhau. Đây là các giải thích có thể, không phải kết luận nhân quả.','',
 '## Một số lỗi nhầm nhiều nhất','',
 '|Nhãn thật|Dự đoán|Số ảnh|','|---:|---:|---:|']
for count,true,pred in confusions: lines.append(f'|{true}|{pred}|{count}|')
lines+=['','Các lỗi trên gợi ý những cặp chữ số có nét gần nhau; muốn giải thích nguyên nhân cụ thể cần xem ảnh sai tại `openml_index` trong test_predictions.csv. Không suy ra cơ chế lỗi chỉ từ confusion matrix, và không dùng các lỗi test để tune lại mô hình trong cùng báo cáo.','',
 '## Quyết định, tính tái lập và giới hạn','',
 '- Năm StratifiedKFold shuffle, seed 202401391; mọi cấu hình dùng lại chính xác các index trong cv_splits.npz.',
 '- Tổng 75 lượt fit CV: 45 warmup + 30 feature engineering. Mỗi fold tạo estimator mới.',
 '- Chuẩn hóa bằng hằng số /255; không fit scaler trên toàn bộ dữ liệu. gamma="scale" chỉ được tính trong fit của từng fold.',
 '- Chốt selection_before_test.json trước khi truy cập test; không tính test riêng cho các baseline/representation khác, tuân thủ yêu cầu test một lần ở phần cuối đề.',
 '- CV được dùng để chọn mô hình nên có thể lạc quan; đây không phải nested CV. SD giữa các fold không phải confidence interval.',
 '- Phiên bản thư viện, seed, checksum train/source, code snapshot, thời gian fit/predict từng fold đều được lưu. Không cần chạy lại test để thảo luận kết quả đã lưu.','',
 '![So sánh CV](cv_comparison.png)','', '![Confusion matrix](confusion_matrix.png)','']
(run/'discussion_vi.md').write_text('\n'.join(lines),encoding='utf-8')
audit={'status':'passed','full_mnist_run':str(run),'cv_fit_rows':len(rows),
 'candidate_count':len(summary['candidates']),'folds':5,
 'each_fold_train':8000,'each_fold_validation':2000,'test_rows':len(predictions),
 'test_correct':correct,'test_accuracy':summary['test_accuracy'],
 'source_matches_snapshot':True,'csv_summary_consistent':True,'saved_test_log_entries':1,
 'analysis_performed_no_fit_or_predict':True,'top_confusions':confusions}
(run/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
(root/'results/latest_mnist_run.txt').write_text(str(run.relative_to(root))+'\n')
print(json.dumps(audit,indent=2))
