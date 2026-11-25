# Mô tả Lab 5 — Nhận diện khuôn mặt thời gian thực với FaceNet & MTCNN trên Webcam

Nguồn: `Chapter3_Feature Detection and Image Matching_Part2.pdf` (slide 8–9).

Notebook: `src/Lab5_Facenet_code.ipynb`.

## Yêu cầu

Pipeline: **Webcam (OpenCV) → MTCNN phát hiện khuôn mặt → FaceNet (`InceptionResnetV1`, pretrained `vggface2`) trích embedding → so cosine similarity** với các khuôn mặt đã đăng ký.

- Ngưỡng của đề bài: similarity **> 0.7** hiển thị **"Matched"**, **< 0.7** hiển thị **"Unknown"**.
- Phát hiện **tất cả** khuôn mặt trong khung hình (`MTCNN(keep_all=True)`).
- Đăng ký người mới: chụp **10 ảnh/người**, lấy trung bình embedding làm đại diện.
- Ảnh chụp khi đăng ký được lưu vào `data/raw/faces/<tên>_<số thứ tự>.jpg`.
- CSDL embedding lưu ở `data/raw/faces/faces_db.pt` — phiên sau nạp lại, không cần chụp lại.
- Hỗ trợ kiểm tra trên **ảnh tĩnh** (mục 4 của notebook) khi không có webcam.

## Cấu trúc phần Lab 5

```text
Lab3/                              # thư mục project hiện tại
└── Lab5/                          # phần FaceNet (Lab 5)
    ├── data/
    │   └── raw/
    │       ├── lena.jpg           # ảnh mẫu cho mục kiểm tra ảnh tĩnh
    │       ├── image.png          # ảnh cá nhân (có thể đổi IMG_PATH sang ảnh này)
    │       └── faces/             # sinh ra khi chạy đăng ký (đã .gitignore)
    │           ├── <tên>_1.jpg … <tên>_10.jpg
    │           └── faces_db.pt    # CSDL embedding
    ├── src/
    │   └── Lab5_Facenet_code.ipynb
    └── Task/
        ├── Lab5_descript.md       # mô tả đề bài (file này)
        └── Lab5_giai_thich.md     # giải thích chi tiết thông số/cách chạy
```

## Cách chạy

1. Cài thư viện:

   ```bash
   pip install opencv-python torch torchvision
   pip install facenet-pytorch
   # Python ≥ 3.13 gặp lỗi pin Pillow cũ:
   # pip install --no-deps facenet-pytorch
   ```

2. Mở `Lab5/src/Lab5_Facenet_code.ipynb`, chạy lần lượt:
   - Ô import + khởi tạo MTCNN/FaceNet (lần đầu tự tải weights ~100MB)
   - Ô test webcam (`q` để thoát)
   - Ô đăng ký `register_face("tên")` — nhấn `s` đủ 10 lần, hoặc `q` để hủy
   - Ô lưu CSDL + ô nạp CSDL (dùng lại ở phiên sau)
   - Ô realtime — nhấn `q` để thoát
3. Không có webcam: chạy mục **4. Kiểm tra trên ảnh tĩnh** (đổi `IMG_PATH` nếu muốn dùng ảnh khác).

## Ghi chú kỹ thuật

- **Embedding 512 chiều**, chuẩn hóa L2 — cosine similarity chỉ phụ thuộc hướng vector.
- Crop khuôn mặt về **160×160** trước khi đưa vào FaceNet (đúng input của vggface2).
- `MTCNN(keep_all=True)` trả về `boxes` + `probs` cho **mọi** khuôn mặt; `probs` là mảng object do interop torch/numpy → phải ép `float`.
- **Đường dẫn tiếng Việt:** `cv2.imread` không đọc được path chứa ký tự Unicode → dùng `imdecode`/`imencode` + `tofile` khi đọc/ghi ảnh.
- **Webcam trên Windows:** dùng `cv2.VideoCapture(idx, cv2.CAP_DSHOW)` để khởi động nhanh; macOS/Linux bỏ `CAP_DSHOW`.
- `THRESHOLD = 0.7` là ngưỡng của đề bài; thực tế nên tinh chỉnh theo điều kiện ánh sáng/thiết bị.
