"""Point operations and linear filters on uint8 grayscale/RGB images."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert('RGB'))


def to_uint8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def brightness(image, beta=40):
    return to_uint8(image.astype(np.float64) + beta)


def contrast(image, alpha=1.5):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    return to_uint8(image.astype(np.float64) * alpha)


def negative(image):
    return 255 - image


def threshold(image, value=128):
    if not 0 <= value <= 255:
        raise ValueError('threshold must lie in [0, 255]')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return np.where(gray >= value, 255, 0).astype(np.uint8)


def check_size(size):
    if not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError('Kernel size must be an odd integer >= 3')


def mean_filter(image, size=5):
    check_size(size)
    kernel = np.ones((size, size), dtype=np.float64) / size**2
    return to_uint8(cv2.filter2D(image.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT_101))


def gaussian_filter(image, size=5, sigma=1.2):
    check_size(size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    vector = cv2.getGaussianKernel(size, sigma)
    kernel = vector @ vector.T
    return to_uint8(cv2.filter2D(image.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT_101))


SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)


def sharpen(image):
    return to_uint8(cv2.filter2D(image.astype(np.float64), -1, SHARPEN_KERNEL, borderType=cv2.BORDER_REFLECT_101))


def process(image):
    return {'original': image, 'brightness_plus40': brightness(image, 40),
            'brightness_minus40': brightness(image, -40),
            'contrast_1.5': contrast(image, 1.5), 'contrast_0.6': contrast(image, 0.6),
            'negative': negative(image), 'threshold_128': threshold(image),
            'mean_5': mean_filter(image), 'gaussian_5_sigma1.2': gaussian_filter(image),
            'sharpen': sharpen(image)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'images' / 'sample.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, image in process(read_rgb(args.input)).items():
        Image.fromarray(image).save(args.output / f'{name}.png')
    print(f'Saved 10 images to {args.output}')


if __name__ == '__main__':
    main()
