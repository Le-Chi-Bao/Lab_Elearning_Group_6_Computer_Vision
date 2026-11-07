"""Chuyển không gian màu; tách dữ liệu HSV/LAB và ảnh dùng để xem."""
import cv2
import numpy as np
from utils import OUTPUTS, read_image, run, save_image, show_opencv


def main(args):
    bgr = read_image(args.image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    save_image("02_grayscale.png", gray)
    # NPY giữ nguyên ý nghĩa các kênh; trình xem PNG thường coi ba kênh là RGB.
    np.save(OUTPUTS / "02_hsv.npy", hsv)
    np.save(OUTPUTS / "02_lab.npy", lab)
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    save_image("02_hsv_to_bgr.png", hsv_bgr)
    save_image("02_lab_to_bgr.png", lab_bgr)
    show_opencv("Grayscale", gray, not args.no_display)
    show_opencv("HSV converted to BGR", hsv_bgr, not args.no_display)
    show_opencv("LAB converted to BGR", lab_bgr, not args.no_display)


if __name__ == "__main__":
    run(main)
