"""Cắt vùng trung tâm và thay đổi kích thước ảnh."""
import cv2
from utils import read_image, run, save_image


def main(args):
    image = read_image(args.image)
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("Ảnh cần rộng và cao ít nhất 2 pixel để crop và resize 50%.")
    # Tọa độ nằm trong ảnh; slicing dùng thứ tự [y1:y2, x1:x2].
    x1, y1 = width // 4, height // 4
    x2, y2 = width - width // 4, height - height // 4
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Vùng crop không hợp lệ.")
    cropped = image[y1:y2, x1:x2].copy()
    fixed = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    # Cùng hệ số cho hai chiều; sai số tối đa do làm tròn pixel.
    scaled = cv2.resize(image, (max(1, round(width * 0.5)), max(1, round(height * 0.5))),
                        interpolation=cv2.INTER_AREA)
    save_image("03_crop_center.png", cropped)
    save_image("03_resize_300x300.png", fixed)
    save_image("03_resize_50_percent.png", scaled)
    print(f"Ảnh gốc: {width}x{height}; ảnh 50%: {scaled.shape[1]}x{scaled.shape[0]}")


if __name__ == "__main__":
    run(main)
