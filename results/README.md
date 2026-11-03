# Kết quả các lần chạy

**Đã hoàn tất full run MNIST thật: test accuracy 97,73%, vượt ngưỡng 96%.**

Run chính thức: `launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/`.

- [English submission report: methods, results, discussion, and all fold scores](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_en.md).
- [Báo cáo đầy đủ từng fold](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/report.md).
- [Thảo luận tiếng Việt và biểu đồ](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/discussion_vi.md).
- [Đối chiếu artifact](launches/20261103T115939_full_661/run_20261103T120001_917778Z_full_667/artifact_audit.json): 75 hàng fold, 15 cấu hình, 5 fold dùng chung, 10.000 dự đoán test, checksum source khớp.

Lựa chọn cuối: SVC RBF C=30, gamma="scale", deskew 784 chiều, mean CV 97,58%. Test chỉ được đánh giá một lần sau khi chốt lựa chọn bằng CV. Các bước phân tích sau đó chỉ đọc kết quả đã lưu, không fit/predict lại.

|Thư mục|Mục đích|Kết quả|
|---|---|---|
|`launches/20261103T115939_full_661/`|Full run MNIST thật|Hoàn tất, exit code 0; test accuracy 97,73%|
|`launches/smoke_20261103T113831Z.log`|Lần tải MNIST đầu, trước khi sửa mạng|HTTP 403 tại api.openml.org; giữ nguyên traceback|
|`validation_synthetic/`|Kiểm tra đầy đủ CV với load_digits phóng/padding thành 28×28|75 lượt fit, 15 cấu hình, cùng 5 fold; không truy cập test|
|`validation_final_artifacts/`|Kiểm tra riêng nhánh refit và ghi output bằng hai ảnh nhân tạo, **CV scores được stub**|Một fit 10.000 mẫu và một predict 10.000 mẫu; selection được lưu trước predict; CSV/confusion matrix/joblib đọc lại được|
|`validation/`|Đề gốc, log và harness tái lập các kiểm tra offline|Các kiểm tra offline qua; không thay thế thực nghiệm MNIST|

Điểm 100% trong kiểm tra output là trên **hai ảnh nhân tạo lặp lại**, chỉ dùng để xác minh cách lưu file. Không có ý nghĩa về chất lượng mô hình hoặc tiêu chí accuracy của đề.

Mỗi full run thật trong tương lai tạo thư mục riêng, lưu source snapshot, cấu hình, điểm từng fold, summary/report, model và dự đoán test. Không ghi đè kết quả trước đó. Xem README ở thư mục gốc để chạy và lưu cả lỗi bootstrap.
