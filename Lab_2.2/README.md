# Lab 2.2 — Toán tử điểm ảnh và lọc tuyến tính

Thực hiện đầy đủ bảy nội dung: thay đổi độ sáng, thay đổi độ tương phản, âm bản, cắt ngưỡng, lọc trung bình, Gaussian và làm sắc nét.

## Chạy chương trình

Mở terminal tại `Lab_2.2`, dùng Python 3.10 trở lên:

```powershell
python -m pip install -r requirements.txt
python image_processing.py
python -m unittest -v
```

Thay ảnh bằng `python image_processing.py --input "duong/dan/anh.jpg" --output outputs/custom`.
Ảnh mẫu được chép từ bài Lab 1 trong repository. Ảnh đọc bằng Pillow để hỗ trợ đường dẫn Unicode, chuyển RGB trước khi xử lý. Kết quả lưu PNG vào `outputs/`.

## Notebook trực quan

Mở `Lab_2_2_Flow.ipynb` bằng VS Code/Jupyter, chọn kernel đã cài requirements rồi **Run All**. Notebook minh họa ảnh gốc, đồ thị ánh xạ mức xám, histogram, kernel, kết quả từng phép xử lý và so sánh ảnh nhiễu trước/sau lọc. Các hàm được dùng lại từ `image_processing.py`.

## Công thức và quy ước

| Phép xử lý | Công thức / cấu hình mặc định |
| --- | --- |
| Độ sáng | g = clip(f + beta), beta = ±40 |
| Tương phản | g = clip(alpha × f), alpha = 1.5 hoặc 0.6 |
| Âm bản | g = 255 − f |
| Cắt ngưỡng | grayscale >= 128 → 255; còn lại → 0 |
| Trung bình | kernel 5×5, mỗi phần tử = 1/25 |
| Gaussian | kernel 5×5, sigma = 1.2, tổng trọng số = 1 |
| Sắc nét | kernel [[0,-1,0],[-1,5,-1],[0,-1,0]] |

Độ tương phản ở đây nhân trực tiếp theo yêu cầu đề; không trừ trung bình hoặc dùng cân bằng histogram. Ngưỡng dùng quy ước **>=**. Các phép màu thực hiện riêng từng kênh RGB; riêng cắt ngưỡng chuyển grayscale trước.

Tính toán bằng float trước khi làm tròn và clip [0,255], tránh tràn số uint8. Lọc dùng biên phản chiếu `BORDER_REFLECT_101`. Các kernel đều đối xứng nên phép correlation của `filter2D` cho kết quả như convolution. Lọc là tuyến tính ở bước tính tổng có trọng số; bước clip/làm tròn đầu ra không tuyến tính.

Trung bình/Gaussian làm mượt nhưng mất chi tiết; sắc nét có thể khuếch đại nhiễu và gây clipping. Các hình minh họa mô tả hành vi, không phải kết luận rằng một phép lọc luôn tốt hơn phép khác.
