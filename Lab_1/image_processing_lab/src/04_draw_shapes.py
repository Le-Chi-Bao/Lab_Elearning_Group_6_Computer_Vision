"""Vẽ trên bản sao; Pillow hỗ trợ văn bản Unicode với font TrueType."""
import os
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from utils import OUTPUTS, read_image, run, save_image


def vietnamese_font(size):
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise OSError("Không tìm thấy font hỗ trợ tiếng Việt. Cập nhật candidates trong 04_draw_shapes.py.")


def main(args):
    original = read_image(args.image)
    height, width = original.shape[:2]
    drawn = original.copy()
    thickness = max(1, min(width, height) // 150)
    start = (width // 10, height // 5)
    end = (width * 9 // 10, height // 5)
    center = (width // 2, height // 2)
    radius = max(1, min(width, height) // 6)
    corner1 = (width // 10, height // 3)
    corner2 = (width * 9 // 10, height * 4 // 5)
    # Màu OpenCV theo thứ tự BGR.
    cv2.line(drawn, start, end, (0, 0, 255), thickness)
    cv2.circle(drawn, center, radius, (0, 255, 0), thickness)
    cv2.rectangle(drawn, corner1, corner2, (255, 0, 0), thickness)
    cv2.putText(drawn, "OpenCV - Image Lab", (width // 20, max(1, height // 10)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.1, min(width / 500, height / 300)),
                (0, 255, 255), thickness, cv2.LINE_AA)
    save_image("04_draw_opencv.png", drawn)
    # Chuyển bản sao BGR sang RGB trước khi đưa vào Pillow.
    pillow_image = Image.fromarray(cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pillow_image)
    draw.line([start, end], fill="red", width=thickness)
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 outline="green", width=thickness)
    draw.rectangle([corner1, corner2], outline="blue", width=thickness)
    font = vietnamese_font(max(8, min(width // 22, height // 12)))
    draw.text((width // 20, height // 30), "Thực hành xử lý ảnh", font=font,
              fill="yellow", stroke_width=1, stroke_fill="black")
    pillow_image.save(OUTPUTS / "04_draw_pillow_vietnamese.png")
    print(f"Đã lưu: {OUTPUTS / '04_draw_pillow_vietnamese.png'}")


if __name__ == "__main__":
    run(main)
