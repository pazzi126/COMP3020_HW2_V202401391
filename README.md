# COMP3020 HW2 — MNIST, Section 6

File nộp bài duy nhất: **`mnist_assignment.py`**. Phần đầu giữ nguyên toàn bộ đoạn tải/chia dữ liệu của đề. Các file trong `results/` là bằng chứng chạy và tài liệu để thảo luận, không phải các file Python cần nộp thêm.

**English assignment report:** [discussion_en.md](results/launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_en.md), including experimental decisions, feature definitions, analysis, final test accuracy, and all 75 fold scores. Code comments and experiment-generated `report.md` are also in English.

## Cài đặt và chạy

Python 3.12; các phiên bản hiện có trong môi trường:

```bash
python3 -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 pandas==2.2.3 joblib==1.5.3 threadpoolctl==3.6.0
```

Chạy từ `/workspace/COMP3020_HW2_V202401391`. OpenML cần quyền truy cập mạng ở lần tải đầu; sklearn sẽ cache dữ liệu. Lệnh dưới lưu cả lỗi tải dữ liệu xảy ra **trước** khi đoạn code bổ sung được thực thi:

```bash
mkdir -p results/launches
launch="results/launches/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$launch"
set -o pipefail
python3 -u mnist_assignment.py --jobs 4 --results-dir "$launch" 2>&1 | tee "$launch/console.log"
status=${PIPESTATUS[0]}
echo "$status" > "$launch/exit_code.txt"
```

Để kiểm tra luồng chạy nhanh, thêm `--smoke` vào lệnh Python: chỉ dùng 300 ảnh từ tập train, vẫn chạy đầy đủ 75 lượt fit CV, **không đánh giá test**. Không dùng điểm smoke để báo cáo bài tập.

Mỗi lần chạy tạo thư mục mới, không ghi đè kết quả cũ. Các mốc thời gian trong tên thư mục là UTC. Không cần chạy dịch vụ nền hay tạo Git worktree.

## Cách đáp ứng criteria

|Yêu cầu|Triển khai|
|---|---|
|10.000 mẫu train đầu, 10.000 mẫu test cuối|Giữ nguyên đoạn mã đề; full run kiểm tra shape 70.000 × 784|
|Tự viết 5/10-fold CV|Vòng lặp 5-fold `StratifiedKFold`; tạo estimator mới mỗi fold; không dùng lớp tìm kiếm tự động|
|Ít nhất 3 estimator, mỗi loại 3 cấu hình|RBF SVC: C=3/10/30; KNN: k=3/5/7; ExtraTrees: min_samples_leaf=1/2/4|
|Điểm từng fold và mean cho mọi cấu hình|`fold_scores.csv`, `cv_candidates.json`, `report.md`|
|Hai phép biến đổi tự viết|Pooling 2×2 → 196 chiều; deskew bằng moment và nội suy → 784 chiều|
|Cùng fold khi so sánh feature|Tái sử dụng cùng danh sách index; lưu `cv_splits.npz`|
|Dùng estimator tốt nhất warmup|Chọn family theo mean CV raw, tune lại 3 cấu hình cho từng representation|
|Không test leakage|Feature chỉ phụ thuộc từng ảnh; chuẩn hóa hằng số /255; gamma của SVC được fit riêng mỗi fold|
|Refit toàn bộ train; test một lần|Chốt và lưu lựa chọn trước khi transform/predict test; một lệnh predict test duy nhất|
|Reproducibility|Seed 202401391, version, checksum dữ liệu train/source, snapshot code, split index|
|Giải thích lựa chọn và quan sát|Comment chi tiết trong code; báo cáo tự sinh dựa trên điểm thực tế|
|Accuracy test ≥96%|Mục tiêu cần đo bằng full run; chương trình ghi rõ đạt/chưa đạt, không tự điều chỉnh theo test|

Đề có hai chỉ dẫn về test: warmup yêu cầu test model thắng, nhưng phần sau yêu cầu chỉ đánh giá test **một lần** cho model cuối. Code tuân theo chỉ dẫn sau: so sánh warmup/feature bằng CV và chỉ tính test cho lựa chọn cuối. Không báo điểm test cho từng representation.

Tổng cộng **75 lượt fit CV**: 45 cho warmup (3 × 3 × 5), thêm 30 cho feature engineering (2 × 3 × 5). Điểm raw của family thắng được tái sử dụng. Full run thêm một lượt fit trên 10.000 mẫu train và một lượt predict test. SVC có thể tốn vài phút tùy máy; code giới hạn cache 512 MB và mặc định tối đa 4 luồng cho các thư viện hỗ trợ.

## Nội dung kết quả

- `metadata.json`: grid, seed, phiên bản, checksum, số mẫu và số luồng.
- `cv_splits.npz`: index train/validation của 5 fold, theo thứ tự tập train dùng trong run.
- `fold_scores.csv`: điểm và thời gian từng fold, flush ngay sau mỗi fold.
- `cv_candidates.json`: kết quả các cấu hình đã hoàn tất; giữ được tiến độ nếu bước sau lỗi.
- `selection_before_test.json`: quyết định đã chốt bằng CV trước test.
- `summary.json`, `report.md`: toàn bộ điểm CV, model tốt nhất, phân tích feature và điểm test nếu là full run.
- `test_predictions.csv`: index OpenML 60000–69999, nhãn thật/dự đoán và đúng/sai.
- `confusion_matrix.csv`: hàng là nhãn thật, cột là dự đoán, thứ tự 0–9.
- `final_model.joblib`: estimator đã refit và tên representation. Khi dùng lại phải chạy đúng chuẩn hóa và hàm feature trong source; chỉ load file joblib đáng tin cậy.
- `source_snapshot.py`, `run.log`, `status.json`: code chính xác của run, log, trạng thái và thời gian chạy.

Sau khi có full run, dùng `report.md` để viết phần thảo luận; đối chiếu thêm confusion matrix và các dự đoán sai. So sánh feature bằng **CV**, không chọn model dựa trên test. SD giữa các fold không phải confidence interval; CV dùng để lựa chọn có thể lạc quan hơn hiệu năng thực tế.

## Trạng thái kiểm chứng hiện tại

Sau khi người dùng cập nhật mạng và xuất bản môi trường, đã tải được MNIST và hoàn tất full run ngày 03/11/2026 (19:00 giờ Việt Nam). Kết quả: **SVC RBF C=30, gamma="scale", deskew 784 chiều; mean CV 97,58%; test accuracy 97,73% (9.773/10.000), vượt ngưỡng 96%**. Mô hình cuối được chọn trước khi đánh giá test đúng một lần. 75 lượt fit CV và các artifact đã được đối chiếu thành công; source hiện tại khớp snapshot của run.

Kết quả chính thức: [report.md](results/launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/report.md), [thảo luận tiếng Việt](results/launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_vi.md). Đường dẫn run gần nhất được ghi tại `results/latest_mnist_run.txt`. Có biểu đồ so sánh CV và confusion matrix ở cả PNG/SVG. Thời gian sau khi tải dữ liệu: khoảng 188 giây.

Lỗi HTTP 403 của lần tải đầu được giữ trong `results/launches/smoke_20261103T113831Z.log` để lưu lịch sử. Không thay đổi hay xuất bản lại cấu hình môi trường trong lần chạy thành công này.

Các kiểm tra offline trước đó dùng `sklearn.datasets.load_digits`, phóng ảnh và padding thành 28×28, được lưu riêng trong `results/validation_synthetic/`. Đây là **dữ liệu kiểm tra tích hợp, không phải MNIST**, không cung cấp accuracy của bài tập. `results/validation/validation_summary.json` ghi các kiểm tra đã hoàn thành. Báo cáo bài tập sử dụng full run MNIST nêu trên.

Nhánh refit/predict và ghi artifact cũng đã được kiểm tra riêng với hai ảnh nhân tạo, lưu trong `results/validation_final_artifacts/`. Các điểm CV trong kiểm tra này được stub để cô lập nhánh xuất kết quả; accuracy của nó không được dùng cho bài tập. Xem `results/README.md` để phân biệt rõ từng loại run.
