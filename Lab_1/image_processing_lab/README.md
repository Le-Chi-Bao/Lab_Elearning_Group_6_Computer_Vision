# Thực hành xử lý ảnh với OpenCV và Pillow

Project dùng Python **3.11 trở lên**, minh họa cài thư viện, đọc và hiển thị ảnh, chuyển không gian màu, crop/resize và vẽ hình. Ảnh gốc luôn được giữ nguyên; kết quả lưu trong `outputs/`, tự tạo nếu chưa tồn tại. Chạy lại sẽ ghi đè các kết quả cùng tên.

## 1. Tạo môi trường ảo

Mở terminal và chuyển vào thư mục project:

```powershell
cd image_processing_lab
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Trên Windows dùng CMD, kích hoạt bằng `.venv\Scripts\activate.bat`. Nếu PowerShell chặn kích hoạt, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` thay cho `python` trong các lệnh dưới đây.

Trên Linux/macOS:

```bash
cd image_processing_lab
python3 -m venv .venv
source .venv/bin/activate
python --version
```

Đảm bảo Python báo phiên bản từ 3.11 trở lên. Nếu máy chưa có Python, cần cài Python trước khi tạo môi trường ảo.

## 2. Cài đặt và kiểm tra thư viện

```bash
python -m pip install -r requirements.txt
python -c "import cv2; import PIL; print('OpenCV:', cv2.__version__); print('Pillow:', PIL.__version__)"
```

`opencv-python` có phần hỗ trợ cửa sổ hiển thị. NumPy được cài theo OpenCV và dùng để đọc dữ liệu ảnh cũng như lưu các kênh HSV/LAB.

## 3. Chuẩn bị ảnh

Tự chép ảnh JPG vào `images/sample.jpg`. Project không tải hoặc tạo ảnh mẫu. Nên dùng ảnh màu rộng và cao ít nhất 300 pixel để dễ quan sát hình vẽ; crop yêu cầu mỗi chiều ít nhất 2 pixel.

```text
image_processing_lab/
├── images/               # Thêm sample.jpg tại đây
├── outputs/
├── src/
│   ├── 01_read_display.py
│   ├── 02_color_conversion.py
│   ├── 03_crop_resize.py
│   ├── 04_draw_shapes.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 4. Chạy từng bài

Có thể chạy toàn bộ bài trong **`image_processing_pipeline.ipynb`**. Notebook chứa code độc lập, giải thích tiếng Việt và hiển thị ảnh ngay dưới từng ô.

Mở notebook bằng VS Code (có extension Jupyter), chọn kernel `.venv` của project rồi chọn **Run All**. Nếu môi trường chưa có kernel:

```powershell
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Ô cài đặt trong notebook dùng `%pip` để cài OpenCV và Pillow vào kernel đang chọn. Đặt ảnh ở `images/sample.jpg`; mở notebook từ project hoặc thư mục cha `Lab1`. Kết quả được lưu với cùng tên như khi chạy các script dưới đây.

Chạy từ thư mục gốc `image_processing_lab/`:

```bash
python src/01_read_display.py
python src/02_color_conversion.py
python src/03_crop_resize.py
python src/04_draw_shapes.py
```

OpenCV chờ một phím trong cửa sổ ảnh để tiếp tục. Pillow dùng trình xem ảnh mặc định của hệ điều hành. Nếu chạy trên máy chủ, terminal không có giao diện hoặc muốn chỉ lưu kết quả:

```bash
python src/01_read_display.py --no-display
python src/02_color_conversion.py --no-display
python src/03_crop_resize.py --no-display
python src/04_draw_shapes.py --no-display
```

Có thể chọn ảnh khác: `python src/01_read_display.py --image images/other.jpg --no-display`. Đường dẫn mặc định được xác định từ vị trí file code nên không phụ thuộc thư mục hiện hành. Đường dẫn tùy chọn tương đối được tính từ thư mục terminal.

Nếu ảnh không tồn tại hoặc không đọc được, chương trình báo lỗi rõ ràng. Nếu OpenCV báo không hỗ trợ GUI, nguyên nhân thường là môi trường không có màn hình hoặc bản thư viện thiếu backend GUI; ảnh đã được lưu trước khi hiển thị. Với môi trường GUI không ổn định, dùng `--no-display`. Pillow cần trình xem ảnh bên ngoài; nếu không thấy cửa sổ, mở trực tiếp PNG trong `outputs/`.

## 5. Giải thích thao tác

- **Đọc và lưu:** OpenCV đọc ảnh màu thành mảng BGR; Pillow chuyển ảnh về RGB. Cả hai lưu JPG đầu vào thành PNG.
- **Grayscale:** chuyển ba kênh màu thành một kênh mức xám.
- **HSV:** biểu diễn sắc độ (H), độ bão hòa (S) và độ sáng (V). Với ảnh uint8 OpenCV, H nằm trong 0–179, S/V trong 0–255.
- **LAB:** biểu diễn độ sáng và hai trục màu. OpenCV mã hóa các kênh vào uint8 khi ảnh đầu vào dùng uint8.
- **Crop:** lấy vùng trung tâm bằng `image[y1:y2, x1:x2]`, kiểm tra kích thước và giới hạn trước khi cắt.
- **Resize cố định:** tạo ảnh 300×300, có thể làm thay đổi tỷ lệ khung hình.
- **Resize 50%:** nhân cả chiều rộng và chiều cao với 0.5 để giữ tỷ lệ, chỉ có sai số làm tròn về pixel nguyên. `INTER_AREA` phù hợp khi thu nhỏ.
- **Vẽ:** OpenCV và Pillow vẽ đường thẳng, hình tròn, hình chữ nhật trên bản sao ảnh. `cv2.putText` minh họa chữ ASCII; Pillow dùng font TrueType để hiển thị tiếng Việt Unicode. Code tìm Arial trên Windows/macOS và DejaVu Sans trên Linux; nếu thiếu font, sửa danh sách `candidates` trong `04_draw_shapes.py` để trỏ tới font hỗ trợ tiếng Việt.

## 6. RGB và BGR

RGB sắp xếp các kênh theo đỏ, xanh lá, xanh dương; BGR theo xanh dương, xanh lá, đỏ. Ví dụ màu đỏ là `(255, 0, 0)` trong RGB nhưng `(0, 0, 255)` trong BGR. Khi chuyển ảnh OpenCV sang Pillow, dùng `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` để tránh đảo màu đỏ và xanh dương.

HSV/LAB không phải RGB/BGR. Không đưa trực tiếp mảng HSV/LAB vào trình xem ảnh màu thông thường. Code chuyển ngược về BGR để lưu PNG và hiển thị OpenCV. Vì vậy các PNG chuyển ngược sẽ gần giống ảnh gốc, có thể lệch nhẹ do lượng tử hóa; chúng không phải ảnh giả màu của các kênh HSV/LAB.

## 7. Kết quả trong outputs/

| File | Nội dung |
| --- | --- |
| `01_opencv.png` | Ảnh PNG lưu bằng OpenCV |
| `01_pillow.png` | Ảnh PNG lưu bằng Pillow |
| `02_grayscale.png` | Ảnh mức xám |
| `02_hsv.npy`, `02_lab.npy` | Dữ liệu kênh HSV/LAB thực, đọc bằng `numpy.load()` |
| `02_hsv_to_bgr.png` | HSV chuyển ngược về BGR để xem đúng màu |
| `02_lab_to_bgr.png` | LAB chuyển ngược về BGR để xem đúng màu |
| `03_crop_center.png` | Vùng trung tâm |
| `03_resize_300x300.png` | Ảnh 300×300 pixel |
| `03_resize_50_percent.png` | Ảnh thu nhỏ còn 50% mỗi chiều |
| `04_draw_opencv.png` | Hình vẽ và chữ bằng OpenCV |
| `04_draw_pillow_vietnamese.png` | Hình vẽ và chữ tiếng Việt bằng Pillow |
