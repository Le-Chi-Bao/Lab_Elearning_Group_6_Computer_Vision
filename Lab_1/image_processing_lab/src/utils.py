"""Các hàm chung cho bài thực hành (Python 3.11 trở lên)."""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def arguments():
    parser = argparse.ArgumentParser(description="Thực hành xử lý ảnh cơ bản")
    parser.add_argument("--image", type=Path, default=ROOT / "images" / "sample.jpg",
                        help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--no-display", action="store_true",
                        help="Chỉ lưu kết quả, không mở cửa sổ")
    return parser.parse_args()


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}. Hãy thêm ảnh vào images/.")
    # imdecode hỗ trợ đường dẫn chứa tiếng Việt trên Windows.
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh, hãy kiểm tra định dạng: {path}")
    return image


def save_image(name, image):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Không mã hóa được ảnh: {path}")
    path.write_bytes(encoded.tobytes())
    print(f"Đã lưu: {path}")
    return path


def show_opencv(title, image, enabled=True):
    if not enabled:
        return
    # Tránh gọi GUI trên Linux không có display (có thể làm tiến trình thoát).
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("Không mở OpenCV: môi trường không có màn hình đồ họa. Xem ảnh trong outputs/.")
        return
    try:
        cv2.imshow(title, image)
        print("Nhấn phím bất kỳ trong cửa sổ ảnh để tiếp tục.")
        cv2.waitKey(0)
    except cv2.error as error:
        print(f"Không mở được cửa sổ OpenCV: {error}. Xem ảnh trong outputs/.")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def show_pillow(image, enabled=True):
    if not enabled:
        return
    try:
        image.show()  # Pillow mở trình xem ảnh của hệ điều hành.
    except (OSError, RuntimeError) as error:
        print(f"Không mở được trình xem Pillow: {error}. Xem ảnh trong outputs/.")


def run(main):
    try:
        main(arguments())
    except (OSError, ValueError) as error:
        raise SystemExit(f"Lỗi: {error}") from None
