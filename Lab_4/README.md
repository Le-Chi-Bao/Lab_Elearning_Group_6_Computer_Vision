# Lab 4 — So sánh ảnh bằng Wavelet Hash

Project thực hành Python theo đề: chuẩn bị ảnh, trích xuất Wavelet, lượng tử hóa thành mã băm, so sánh Hamming, đánh giá và tìm kiếm ảnh tương tự.

Notebook trực quan: [Wavelet_Flow.ipynb](Wavelet_Flow.ipynb). Mở bằng VS Code (Python + Jupyter) hoặc JupyterLab, chọn kernel Python và chạy **Run All**. Notebook minh họa ảnh đầu vào, các thành phần Wavelet, LL, hash, XOR/Hamming, bảng đánh giá, ROC, ma trận nhầm lẫn và top-k. Có cell cài thư viện; kết quả sinh vào `outputs/notebook/`. Notebook chưa có output được thực thi sẵn.

## 1. Cài đặt và chạy

Mở PowerShell tại thư mục `Lab_4`, dùng Python 3.10 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe wavelet_lab.py prepare-demo
.\.venv\Scripts\python.exe wavelet_lab.py evaluate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Không bắt buộc activate môi trường. `prepare-demo` lấy hai ảnh mẫu đi kèm scikit-learn, không tải dữ liệu qua mạng. Cài thư viện lần đầu vẫn cần mạng hoặc wheel cục bộ.

## 2. Cấu trúc

```text
Lab_4/
  wavelet_lab.py       # CLI, thuật toán, dữ liệu, đánh giá, tìm kiếm
  requirements.txt
  tests/test_wavelet.py
  data/demo/          # tạo khi chạy prepare-demo
  outputs/           # tạo khi chạy evaluate/search
```

## 3. Phương pháp

1. Áp dụng hướng EXIF, chuyển grayscale, resize 128×128, chuẩn hóa về [0,1].
2. Phân rã Wavelet 4 mức bằng PyWavelets với chế độ `periodization`. Lấy hệ số xấp xỉ LL kích thước 8×8, giúp ba họ Wavelet có cùng độ dài hash.
3. Lượng tử hóa theo median: hệ số lớn hơn median thành 1, còn lại thành 0. Trải phẳng thành 64 bit.
4. Khoảng cách Hamming chuẩn hóa = số bit khác nhau / 64. Giá trị càng nhỏ, ảnh càng tương tự. Điểm tương tự = 1 − khoảng cách; đây không phải xác suất.
5. Chọn ngưỡng bằng balanced accuracy trên validation. Dự đoán tương tự khi `distance <= threshold`. Nếu nhiều ngưỡng đồng hạng, chọn ngưỡng nhỏ nhất.
6. Giữ nguyên ngưỡng khi đánh giá test. ROC sử dụng điểm tương tự liên tục, không dùng nhãn dự đoán nhị phân.

Không sao chép trực tiếp mã minh họa trong slide: `pywt.quantize` không phải API được dùng ở đây và các hệ số chi tiết của `wavedec2` là tuple. Project lượng tử hóa LL thành bit rõ ràng và bảo đảm độ dài mã cố định.

## 4. Dữ liệu và ground truth

Demo chia mỗi ảnh mẫu thành sáu crop: tổng 12 nhóm nguồn. Mỗi nhóm có năm phiên bản: gốc, tăng sáng, nhiễu Gaussian seed 42, giảm kích thước và nén JPEG. Có 60 ảnh. Hai ảnh cùng nhóm là cặp tương tự; khác nhóm là cặp không tương tự. Mỗi split có 30 ảnh: 60 cặp dương và 375 cặp âm.

Ba crop đầu của mỗi ảnh dùng cho validation; ba crop còn lại cho test. Mọi biến thể của cùng crop nằm trong một split để tránh rò rỉ nhóm nguồn. Tuy nhiên các crop vẫn xuất phát từ hai ảnh chụp; kết quả demo không đại diện cho khả năng tổng quát hóa trên ảnh thực tế. Demo chưa kiểm chứng thay đổi góc chụp hay nhận diện cùng đối tượng ở nhiều bối cảnh.

Để đánh giá nghiêm túc, dùng nhiều đối tượng độc lập, ảnh chụp nhiều góc, nhiều mức nhiễu và nhóm âm có hình thức gần giống nhau. Chia validation/test theo đối tượng hoặc ảnh nguồn trước khi sinh biến thể. Không điều chỉnh thuật toán theo kết quả test.

Tạo manifest CSV UTF-8 với đường dẫn tương đối so với vị trí manifest:

```csv
path,group,split
images/object_a_1.jpg,object_a,validation
images/object_a_2.jpg,object_a,validation
images/object_b_1.jpg,object_b,validation
images/object_c_1.jpg,object_c,test
images/object_c_2.jpg,object_c,test
images/object_d_1.jpg,object_d,test
```

```powershell
.\.venv\Scripts\python.exe wavelet_lab.py evaluate --manifest data/real/manifest.csv --output outputs/real
```

Mỗi split cần cả cặp tương tự và không tương tự. Nhóm không được xuất hiện ở cả hai split. Chương trình kiểm tra hai điều kiện này. Toàn bộ cặp trong từng split được sử dụng; chi phí đánh giá tăng theo O(N²).

## 5. Kết quả đánh giá

`outputs/metrics.csv` và `metrics.json` chứa chỉ số cho Haar, db2, db4:

| Chỉ số | Công thức / ý nghĩa |
| --- | --- |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) |
| Sensitivity / Recall | TP/(TP+FN) |
| Specificity | TN/(TN+FP) |
| Precision | TP/(TP+FP), quy ước bằng 0 khi không có dự đoán dương |
| ROC AUC | Diện tích dưới ROC, sử dụng 1 − distance |

Dương nghĩa là cặp ảnh tương tự. **Specificity khác precision**; phần khung code của slide dùng tên precision cho độ đặc hiệu chưa chính xác.

Các tệp bổ sung:

- `roc.png`: so sánh ROC của ba phương pháp trên test.
- `confusion_haar.png`, `confusion_db2.png`, `confusion_db4.png`: ma trận nhầm lẫn tại ngưỡng validation.
- `pairs_*.csv`: từng cặp test, nhãn thật và khoảng cách để kiểm tra lỗi phân loại.

Không đưa số liệu giả định vào báo cáo: cần chạy chương trình rồi lấy số liệu từ các tệp trên. Khi phân tích, xem đồng thời recall, specificity và AUC vì số cặp âm nhiều hơn cặp dương. Các cặp dùng chung ảnh không độc lập thống kê.

## 6. Ứng dụng tìm kiếm ảnh

```powershell
.\.venv\Scripts\python.exe wavelet_lab.py search --query data/demo/photo0_crop0_original.jpg --directory data/demo --wavelet haar --top-k 5
```

Ứng dụng dòng lệnh duyệt ảnh trong thư mục con, loại chính tệp truy vấn, xếp theo Hamming tăng dần và xuất `outputs/search.csv`. Có thể dùng ảnh truy vấn bên ngoài dataset. Các ảnh trùng nội dung nhưng có tên khác vẫn được trả về. Ảnh lỗi sẽ làm lệnh dừng để người dùng xử lý dữ liệu.

## 7. Gợi ý trình bày báo cáo

Nêu mục tiêu; mô tả nguồn ảnh và quy tắc gán nhãn; giải thích LL và lượng tử hóa; ghi cấu hình, seed, cách chia tập và chọn ngưỡng; chèn bảng metrics, ROC và confusion matrix; phân tích cặp sai từ CSV; minh họa top-k; kết luận theo số liệu thực tế.

Wavelet hash phù hợp tìm ảnh gần trùng, nhưng có thể nhạy với xoay, crop, thay đổi bố cục; không thay thế mô hình nhận diện ngữ nghĩa. Có thể khảo sát thêm độ dài hash hoặc kết hợp hệ số chi tiết, với ngưỡng được chọn lại trên validation.
