"""Đọc, hiển thị và chuyển định dạng ảnh bằng OpenCV và Pillow."""
from PIL import Image
from utils import OUTPUTS, read_image, run, save_image, show_opencv, show_pillow


def main(args):
    bgr = read_image(args.image)
    save_image("01_opencv.png", bgr)
    with Image.open(args.image) as source:
        rgb = source.convert("RGB")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rgb.save(OUTPUTS / "01_pillow.png")
    print(f"Đã lưu: {OUTPUTS / '01_pillow.png'}")
    show_opencv("OpenCV - BGR", bgr, not args.no_display)
    show_pillow(rgb, not args.no_display)


if __name__ == "__main__":
    run(main)
