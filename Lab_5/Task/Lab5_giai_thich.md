# GIẢI THÍCH CHI TIẾT LAB 5 — để trả lời khi bị hỏi

Tài liệu này giải thích **tất cả thông số, cách chạy và file được tạo ra** bởi
notebook `src/Lab5_Facenet_code.ipynb` (nhận diện khuôn mặt realtime bằng FaceNet + MTCNN).

> Đường dẫn trong tài liệu tính từ thư mục `Lab5/`.

---

# PHẦN B — LAB5_FACENET_CODE.IPYNB (Nhận diện khuôn mặt realtime)

## B1. Pipeline (nói 10 giây)

> "Webcam OpenCV → MTCNN phát hiện khuôn mặt → crop 160×160 → FaceNet
> (InceptionResnetV1 pretrained 'vggface2') → embedding 512 chiều đã chuẩn hóa
> → cosine similarity với các khuôn mặt đã đăng ký → **> 0.7: Matched, < 0.7: Unknown**."

## B2. Thông số chính

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| `THRESHOLD` | **0.7** | Ngưỡng của đề bài: similarity > 0.7 → "Matched", ngược lại "Unknown" |
| `device` | `cuda` nếu có, không thì `cpu` | Tính toán trên GPU/CPU tự động |
| `MTCNN(keep_all=True)` | — | Phát hiện **tất cả** khuôn mặt trong frame (không chỉ 1) |
| Kích thước crop | 160×160 | FaceNet vggface2 yêu cầu đầu vào cố định 160×160 |
| Embedding | **512 chiều**, chuẩn hóa L2 (norm = 1) | Khuôn mặt giống nhau → vector gần nhau; chuẩn hóa để cosine chỉ phụ thuộc góc |
| `n_shots` | 10 ảnh/người | Đăng ký: chụp 10 lần rồi **lấy trung bình embedding** → giảm nhiễu |
| Chỉ số so sánh | **Cosine similarity** ∈ [−1, 1] | Chọn người có similarity cao nhất trong CSDL |

💡 **Ví dụ "thông số đại diện cho cái gì" khi bị hỏi:**
- **THRESHOLD = 0.7:** mình đứng trước camera → sim với "ten_than" = 0.85 > 0.7 → hiện `Matched: ten_than (0.85)`; người lạ đứng vào → sim cao nhất chỉ 0.62 → `Unknown (0.62)`. Ngưỡng này của **đề bài**, thực tế tinh chỉnh theo bộ test.
- **Embedding 512 chiều:** mỗi khuôn mặt = 1 điểm trong không gian 512 chiều; 2 ảnh cùng người → 2 điểm sát nhau (cosine ≈ 0.8–0.9), khác người → xa nhau (≈ 0.2–0.5). Chuẩn hóa L2 (norm = 1) giúp cosine chỉ đo **hướng** vector, không đo độ dài.
- **MTCNN(keep_all=True):** 2 người cùng đứng trong 1 khung hình → 2 box, 2 embedding, 2 nhãn trên cùng 1 frame. `prob` (vd 0.9999) là độ tin cậy "đây là mặt người" của detector.
- **Crop 160×160:** FaceNet vggface2 học trên ảnh 160×160 — cần resize mọi khuôn mặt về cùng kích thước này trước khi đưa vào mạng.
- **n_shots = 10:** mỗi lần nhấn `s` được 1 embedding (512 chiều); lấy **trung bình 10 vector** cho "ten_than" → vector đại diện ổn định hơn chụp 1 lần (giảm nhiễu ánh sáng/góc nhìn).
- **Nguồn so khớp:** vec mới từ camera so với **tất cả** người trong `FACE_DB` bằng cosine một lần (1 phép nhân ma trận), lấy argmax.

## B3. Các hàm chính

| Hàm | Việc gì |
|---|---|
| `open_webcam(idx)` | Mở camera (`CAP_DSHOW` — khởi động nhanh trên Windows); assert nếu không mở được |
| `detect_faces_bgr(frame)` | `mtcnn.detect` trên frame đã đổi BGR→RGB; trả [(box, prob), ...]; **prob là mảng object do interop torch/numpy** → ép float |
| `embed_faces(frame, boxes)` | Crop từng box → resize 160×160 → RGB → tensor [0,1] → FaceNet → (N, 512) |
| `register_face(name, n_shots)` | Vòng lặp chụp: nhấn **`s`** để chụp (đếm x/n_shots trên màn hình), **`q`** hủy; lưu ảnh + trả embedding trung bình |
| `match_face(vec, db)` | Cosine similarity giữa vec và toàn bộ CSDL → (tên, sim) cao nhất |

## B4. File được tạo ra khi chạy FaceNet

| File | Sinh khi nào | Nội dung |
|---|---|---|
| `data/raw/faces/<tên>_<n>.jpg` | Mỗi lần nhấn `s` trong `register_face` | **Frame chụp** (ảnh full khung hình camera) |
| `data/raw/faces/faces_db.pt` | Ô "Lưu CSDL" sau khi đăng ký xong | Dictionary `{tên: tensor 512 chiều}` — phiên sau chạy ô "nạp CSDL" là nhận diện được **không cần chụp lại** |

💡 **Ví dụ thực tế trên đĩa (như hiện tại trong lab):**
```
data/raw/faces/
├── ten_than_1.jpg … ten_than_10.jpg    # 10 frame chụp người thứ nhất
├── nguoi_hai_1.jpg … nguoi_hai_10.jpg  # 10 frame chụp người thứ hai
└── faces_db.pt                         # {'ten_than': tensor(512,), 'nguoi_hai': tensor(512,)}
```
Phiên sau: chỉ chạy ô **"nạp CSDL đã lưu"** → `FACE_DB` đầy đủ, chờ realtime luôn — không cần chụp lại 2×10 ảnh.
Lưu ý: thư mục `data/raw/faces/` đã nằm trong `.gitignore` (không push ảnh cá nhân lên git).

## B5. Các nhãn hiển thị khi chạy realtime

- Khung **xanh lá** + `Matched: <tên> (0.xx)` khi sim > 0.7
- Khung **đỏ** + `Unknown (0.xx)` khi sim ≤ 0.7
- Không phát hiện khuôn mặt: chữ "Khong phat hien khuon mat" (đỏ)
- Nhấn `q` để thoát, `s` để chụp khi đăng ký

## B6. Trả lời nhanh những câu hay bị hỏi

1. **Vì sao FaceNet không cần huấn luyện?** — Lấy weights đã train sẵn trên ~3 triệu ảnh khuôn mặt (vggface2); chỉ dùng để **trích đặc trưng**, phần so khớp là phép tính cosine.
2. **Vì sao chọn 0.7?** — Là ngưỡng mặc định của đề bài; thực tế nên tinh chỉnh theo bộ test riêng (mờ, nghiêng, che khuất → hạ ngưỡng có thể tăng recall nhưng tăng nhầm).
3. **Vì sao đăng ký 10 ảnh rồi lấy trung bình?** — Mỗi frame có nhiễu; trung bình 10 embedding ổn định hơn 1 ảnh (kỹ thuật enrollment tiêu chuẩn).
4. **Hạn chế?** — Nhạy độ phân giải thấp/góc nghiêng lớn/che khuất; người ngoài CSDL luôn Unknown; chưa có chống giả mạo (ảnh in, video replay).
5. **Có cần webcam không?** — Không bắt buộc: mục 4 của notebook kiểm tra trên **ảnh tĩnh** (`IMG_PATH`, mặc định `data/raw/lena.jpg`; đổi sang `data/raw/image.png` nếu muốn).

---

# PHẦN C — SỐ LIỆU NHANH (thuộc lòng để trả lời)

- FaceNet: embedding **512 chiều**, ngưỡng **0.7**, crop **160×160**, đăng ký **10 ảnh/người**.
- Model: `InceptionResnetV1(pretrained="vggface2")` + `MTCNN(keep_all=True)`.
- CSDL hiện có: **2 người** (`ten_than`, `nguoi_hai`), mỗi người **1 vector 512 chiều** (trung bình 10 lần chụp).
- File: ảnh chụp ở `data/raw/faces/<tên>_<n>.jpg`, CSDL ở `data/raw/faces/faces_db.pt` (đã `.gitignore`).
