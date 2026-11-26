# Lab 2 — Xử lý ảnh với OpenCV

> Bài thực hành xử lý ảnh cơ bản sử dụng **OpenCV**, **NumPy**, **Matplotlib**, **Pillow** trong môi trường ảo (venv).

---

## 📌 Mục tiêu bài thực hành

1. **Toán tử điểm ảnh (Point Operations)** — thay đổi độ sáng, độ tương phản, âm bản, cắt ngưỡng.
2. **Lọc tuyến tính (Linear Filters)** — lọc trung bình, Gaussian, làm sắc nét.
3. **Bài tập nâng cao** — phát hiện cạnh (Sobel/Prewitt), kernel tự thiết kế, so sánh các loại lọc, lọc phi tuyến (Median/Bilateral).

---

## 🖼️ Ảnh đầu vào
C:\Users\asus\Downloads\anhne.jpg

> ⚠️ Nếu đường dẫn ảnh khác, sửa biến `img_path` trong `lab2.ipynb` cho đúng.

---

## 📂 Cấu trúc project

---

## ⚙️ Hướng dẫn cài đặt

### 1. Tạo môi trường ảo (venv)

Mở Terminal trong VS Code tại folder `lab2`, chạy:

```bash
python -m venv venv