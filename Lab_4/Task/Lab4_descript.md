# Mô tả Lab 4 — So sánh sự tương đồng của các hình ảnh sử dụng Wavelet (Python)

Nguồn: `Chapter3_Feature Detection and Image Matching_Part1.pdf` (slide 16–18).

Notebook: `src/Lab4_code.ipynb`.

## Đề bài

1. **Chuẩn bị dữ liệu:** tập ảnh gồm các cặp tương tự (cùng đối tượng, góc/mức nhiễu khác nhau) và cặp không tương tự.
2. **Trích xuất wavelet:** dùng biến đổi wavelet chuyển từng ảnh thành ma trận wavelet.
3. **Tạo mã băm** cho mỗi ảnh dựa trên các hệ số wavelet đã lượng tử hóa.
4. **So sánh hash:** tính khoảng cách Hamming giữa các mã băm để đánh giá độ tương đồng.
5. **Đánh giá:** accuracy, recall (sensitivity), specificity (độ đặc hiệu), precision, đường cong ROC.

Bài tập nâng cao:

6. Khảo sát các phương pháp băm wavelet khác nhau và so sánh hiệu suất.
7. Xây dựng ứng dụng tìm kiếm hình ảnh dựa trên hàm băm wavelet.

## Cấu trúc phần Lab 4

```text
Lab3/                          # thư mục project hiện tại
└── Lab4/                      # phần Wavelet (Lab 4)
    ├── data/
    │   ├── raw/               # 10 ảnh gốc tải từ bộ mẫu OpenCV
    │   └── processed/
    │       ├── images/        # 60 ảnh: 10 gốc 256×256 + 50 ảnh biến đổi
    │       └── pairs.csv      # 100 cặp: 50 dương (label=1) / 50 âm (label=0)
    ├── outputs/               # hình (fig_01..fig_07) + variant_results.csv
    ├── src/
    │   ├── Lab4_code.ipynb    # notebook chính: 5 task + 2 task nâng cao
    │   ├── wavelet_hash.py    # hàm hash / hamming / evaluate / pick_threshold
    │   └── prepare_data.py    # tải ảnh, sinh ảnh biến đổi và bộ cặp (seed cố định)
    └── Task/
        ├── Lab4_descript.md   # mô tả đề bài (file này)
        └── Lab4_giai_thich.md # giải thích chi tiết thông số/bảng/biểu đồ
```

## Cách chạy

1. Từ thư mục gốc project: `pip install -r requirements.txt`
2. Mở `Lab4/src/Lab4_code.ipynb` → **Run All** (chạy từ trên xuống).
   Lần đầu notebook tự tải ảnh và sinh dataset (seed cố định, tái lập được).

## Ghi chú kỹ thuật so với pseudocode trong slide

- `pywt.wavedec2` trả về **tuple lồng nhau** (LL là mảng, detail là các tuple) — không flatten trực tiếp được; bản này chỉ dùng subband LL.
- Ảnh phải **resize về kích thước cố định (128×128)** trước khi hash, nếu không LL hai ảnh khác kích thước → không so Hamming được.
- `pywt.quantize` không có trong PyWavelets hiện hành — bản so sánh dùng `np.round(coeff / step)`.
- Metric: slide ghi "Độ đặc hiệu: precision" là nhầm — **độ đặc hiệu = specificity = TN/(TN+FP)**, còn precision = TP/(TP+FP).
