"""Canny edge detection: educational stages, libraries, evaluation and video."""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation
from skimage.feature import canny

ROOT = Path(__file__).resolve().parent


def read_gray(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('L'))


def validate(sigma, low, high):
    if not np.isfinite([sigma, low, high]).all() or sigma <= 0 or not 0 <= low < high <= 255:
        raise ValueError('Require sigma > 0 and 0 <= low < high <= 255')


def smooth(gray, sigma):
    return cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)


def opencv_canny(gray, sigma=1, low=50, high=100):
    validate(sigma, low, high)
    blurred = np.clip(np.rint(smooth(gray, sigma)), 0, 255).astype(np.uint8)
    return cv2.Canny(blurred, low, high, apertureSize=3, L2gradient=True) > 0


def skimage_canny(gray, sigma=1, low=50, high=100):
    validate(sigma, low, high)
    return canny(gray.astype(float) / 255, sigma=sigma,
                 low_threshold=low / 255, high_threshold=high / 255, mode='reflect')


def manual_canny(gray, sigma=1, low=50, high=100):
    """Teaching version: 4-direction NMS and 8-connected hysteresis."""
    validate(sigma, low, high)
    blurred = smooth(gray, sigma)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    nms = np.zeros_like(magnitude)
    center = magnitude[1:-1, 1:-1]
    a = angle[1:-1, 1:-1]
    directions = [((a < 22.5) | (a >= 157.5), magnitude[1:-1, :-2], magnitude[1:-1, 2:]),
                  ((a >= 22.5) & (a < 67.5), magnitude[:-2, :-2], magnitude[2:, 2:]),
                  ((a >= 67.5) & (a < 112.5), magnitude[:-2, 1:-1], magnitude[2:, 1:-1]),
                  ((a >= 112.5) & (a < 157.5), magnitude[:-2, 2:], magnitude[2:, :-2])]
    for mask, before, after in directions:
        keep = mask & (center >= before) & (center >= after)
        nms[1:-1, 1:-1][keep] = center[keep]
    strong = (nms >= high) & (nms > 0)
    candidates = ((nms >= low) & (nms > 0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(candidates, connectivity=8)
    connected = np.unique(labels[strong])
    connected = connected[connected != 0]
    edges = np.isin(labels, connected)
    return {'blurred': blurred, 'gx': gx, 'gy': gy, 'magnitude': magnitude,
            'angle': angle, 'nms': nms, 'strong': strong,
            'weak': candidates.astype(bool) & ~strong, 'edges': edges}


def synthetic_scene():
    image = np.full((256, 320), 25, dtype=np.uint8)
    cv2.rectangle(image, (25, 35), (130, 140), 220, -1)
    cv2.circle(image, (230, 90), 45, 170, -1)
    triangle = np.array([[170, 220], [240, 155], [295, 225]], np.int32)
    cv2.fillPoly(image, [triangle], 240)
    # Analytic boundaries from the actual geometry, independent of detectors.
    truth = np.zeros_like(image)
    cv2.rectangle(truth, (25, 35), (130, 140), 1, 1)
    cv2.circle(truth, (230, 90), 45, 1, 1)
    cv2.polylines(truth, [triangle], True, 1, 1)
    return image, truth.astype(bool)


def edge_metrics(prediction, truth, tolerance=2):
    if prediction.shape != truth.shape or tolerance < 0:
        raise ValueError('Shapes must agree and tolerance must be nonnegative')
    pred, target = prediction.astype(bool), truth.astype(bool)
    y, x = np.ogrid[-tolerance:tolerance+1, -tolerance:tolerance+1]
    disk = x*x + y*y <= tolerance*tolerance
    precision = float((pred & binary_dilation(target, structure=disk)).sum() / max(pred.sum(), 1))
    recall = float((target & binary_dilation(pred, structure=disk)).sum() / max(target.sum(), 1))
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12)}


def benchmark(gray, truth):
    def sobel():
        source = smooth(gray, 1)
        return np.hypot(cv2.Sobel(source, cv2.CV_64F, 1, 0), cv2.Sobel(source, cv2.CV_64F, 0, 1)) >= 100
    def laplacian():
        return np.abs(cv2.Laplacian(smooth(gray, 1), cv2.CV_64F, ksize=3)) >= 40
    methods = {'OpenCV': lambda: opencv_canny(gray), 'scikit-image': lambda: skimage_canny(gray),
               'Manual': lambda: manual_canny(gray)['edges'], 'Sobel magnitude': sobel,
               'Laplacian magnitude': laplacian}
    rows, predictions = [], {}
    for name, operation in methods.items():
        operation()  # warm-up
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            prediction = operation()
            durations.append((time.perf_counter() - start) * 1000)
        predictions[name] = prediction
        rows.append({'method': name, 'median_ms': float(np.median(durations)),
                     'edge_pixels': int(prediction.sum()), **edge_metrics(prediction, truth)})
    return rows, predictions


def regions_and_shapes(gray):
    edges = opencv_canny(gray)
    closed = cv2.morphologyEx(edges.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    regions = np.zeros_like(gray)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        circularity = 4 * np.pi * area / max(perimeter**2, 1e-12)
        shape = 'triangle' if len(polygon) == 3 else 'quadrilateral' if len(polygon) == 4 else 'circle' if circularity > 0.8 else 'other'
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(overlay, [contour], -1, (255, 70, 20), 2)
        cv2.putText(overlay, shape, (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 20), 1)
        cv2.drawContours(regions, [contour], -1, 255, -1)
        found.append({'shape': shape, 'area': float(area), 'bbox': [x, y, w, h]})
    return closed, regions, overlay, found


def process_video(source, output, sigma=1, low=50, high=100):
    validate(sigma, low, high)
    capture = cv2.VideoCapture(str(source))
    writer = None
    frames = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            edges = opencv_canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), sigma, low, high)
            if writer is None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot initialize mp4v video writer')
            writer.write(cv2.cvtColor(edges.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR))
            frames += 1
        if not frames:
            raise ValueError('Video contains no readable frames')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'images'/'sample.jpg')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--low', type=float, default=50)
    parser.add_argument('--high', type=float, default=100)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.video:
        print('Frames:', process_video(args.video, args.output/'canny_video.mp4', args.sigma, args.low, args.high))
        return
    gray = read_gray(args.input)
    for name, edges in [('opencv', opencv_canny(gray,args.sigma,args.low,args.high)),
                        ('skimage', skimage_canny(gray,args.sigma,args.low,args.high)),
                        ('manual', manual_canny(gray,args.sigma,args.low,args.high)['edges'])]:
        Image.fromarray(edges.astype(np.uint8)*255).save(args.output/f'{name}.png')
    scene, truth = synthetic_scene()
    rows, _ = benchmark(scene, truth)
    (args.output/'benchmark.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
