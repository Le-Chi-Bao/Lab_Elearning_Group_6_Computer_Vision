"""Wavelet image hashing, reproducible evaluation, and image retrieval."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image, ImageEnhance, ImageOps
from sklearn.datasets import load_sample_images
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent


def wavelet_hash(image_path, wavelet="haar", hash_size=8, image_size=128):
    """Return hash_size² bits from median-quantized approximation coefficients."""
    if hash_size < 2 or hash_size & (hash_size - 1):
        raise ValueError("hash_size must be a power of two >= 2")
    if image_size <= hash_size or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two larger than hash_size")
    level = int(np.log2(image_size // hash_size))
    # Periodization preserves a common coefficient shape for every wavelet.
    if level > pywt.dwt_max_level(image_size, pywt.Wavelet(wavelet).dec_len):
        raise ValueError("Wavelet filter too long for requested decomposition")
    with Image.open(image_path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        gray = gray.resize((image_size, image_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64) / 255.0
    approximation = pywt.wavedec2(pixels, wavelet, mode="periodization", level=level)[0]
    return (approximation > np.median(approximation)).ravel()


def hamming_distance(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("Hashes must be nonempty vectors of equal length")
    return float(np.mean(first != second))


def save_csv(destination, rows):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_demo(destination):
    """Make twelve source groups from bundled sklearn photographs (no download)."""
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for photo_id, array in enumerate(load_sample_images().images):
        photo = Image.fromarray(array)
        width, height = photo.size
        for crop_id in range(6):
            col, row = crop_id % 3, crop_id // 3
            base = photo.crop((col * width // 3, row * height // 2,
                               (col + 1) * width // 3, (row + 1) * height // 2))
            base = base.resize((256, 256), Image.Resampling.LANCZOS)
            group = f"photo{photo_id}_crop{crop_id}"
            split = "validation" if crop_id < 3 else "test"
            noisy = np.clip(np.asarray(base).astype(float) + rng.normal(0, 12, (256, 256, 3)), 0, 255)
            variants = {"original": base, "bright": ImageEnhance.Brightness(base).enhance(1.25),
                        "noise": Image.fromarray(noisy.astype("uint8")),
                        "small": base.resize((80, 80)), "jpeg": base}
            for variant, image in variants.items():
                name = f"{group}_{variant}.jpg"
                image.save(destination / name, quality=35 if variant == "jpeg" else 95)
                rows.append({"path": name, "group": group, "split": split})
    save_csv(destination / "manifest.csv", rows)
    print(f"Created {len(rows)} images: {destination / 'manifest.csv'}")


def read_manifest(manifest):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"path", "group", "split"}.issubset(rows[0]):
        raise ValueError("Manifest requires path,group,split columns")
    group_splits = {}
    for row in rows:
        if row["split"] not in {"validation", "test"} or not row["group"]:
            raise ValueError("Each row needs a group and validation/test split")
        previous = group_splits.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("A source group must not occur in both splits")
        row["file"] = manifest.parent / row["path"]
        if not row["file"].is_file():
            raise FileNotFoundError(row["file"])
    return rows


def pair_scores(rows, hashes, split):
    pairs = []
    for i, j in combinations(range(len(rows)), 2):
        if rows[i]["split"] == rows[j]["split"] == split:
            pairs.append({"first": rows[i]["path"], "second": rows[j]["path"],
                          "label": int(rows[i]["group"] == rows[j]["group"]),
                          "distance": hamming_distance(hashes[i], hashes[j])})
    if not pairs or len({p["label"] for p in pairs}) != 2:
        raise ValueError(f"{split} needs both similar and dissimilar pairs")
    return pairs, np.array([p["label"] for p in pairs]), np.array([p["distance"] for p in pairs])


def evaluate(manifest, output):
    rows = read_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for wavelet in ("haar", "db2", "db4"):
        hashes = [wavelet_hash(row["file"], wavelet) for row in rows]
        _, val_y, val_d = pair_scores(rows, hashes, "validation")
        candidates = np.r_[-1e-9, np.unique(val_d)]
        def balanced_accuracy(threshold):
            tn, fp, fn, tp = confusion_matrix(val_y, val_d <= threshold, labels=[0, 1]).ravel()
            return (tp / (tp + fn) + tn / (tn + fp)) / 2
        threshold = float(max(candidates, key=balanced_accuracy))
        pairs, labels, distances = pair_scores(rows, hashes, "test")
        predictions = distances <= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        result = {"wavelet": wavelet, "bits": 64, "threshold": threshold,
                  "accuracy": float(accuracy_score(labels, predictions)),
                  "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
                  "specificity": float(tn / (tn + fp)),
                  "precision": float(precision_score(labels, predictions, zero_division=0)),
                  "roc_auc": float(roc_auc_score(labels, 1 - distances)),
                  "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        results.append(result)
        save_csv(output / f"pairs_{wavelet}.csv", pairs)
        fpr, tpr, _ = roc_curve(labels, 1 - distances)
        ax.plot(fpr, tpr, label=f"{wavelet}: AUC={result['roc_auc']:.3f}")
        matrix_fig, matrix_ax = plt.subplots(figsize=(4, 4))
        matrix_ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
        for (r, c), value in np.ndenumerate([[tn, fp], [fn, tp]]):
            matrix_ax.text(c, r, str(value), ha="center", va="center")
        matrix_ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Different", "Similar"],
                      yticklabels=["Different", "Similar"], xlabel="Prediction", ylabel="Ground truth",
                      title=f"{wavelet}, threshold={threshold:.3f}")
        matrix_fig.tight_layout()
        matrix_fig.savefig(output / f"confusion_{wavelet}.png", dpi=160)
        plt.close(matrix_fig)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Wavelet hash ROC (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "roc.png", dpi=160)
    plt.close(fig)
    save_csv(output / "metrics.csv", results)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def search(query, directory, wavelet, top_k, output):
    if top_k < 1:
        raise ValueError("top-k must be positive")
    query_hash = wavelet_hash(query, wavelet)
    matches = []
    for file in sorted(directory.rglob("*")):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and file.resolve() != query.resolve():
            distance = hamming_distance(query_hash, wavelet_hash(file, wavelet))
            matches.append({"path": str(file), "distance": distance, "similarity": 1 - distance})
    matches.sort(key=lambda item: (item["distance"], item["path"]))
    if not matches:
        raise ValueError("No candidate images found")
    save_csv(output, matches[:top_k])
    print(json.dumps(matches[:top_k], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("prepare-demo")
    demo.add_argument("--output", type=Path, default=ROOT / "data" / "demo")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--manifest", type=Path, default=ROOT / "data" / "demo" / "manifest.csv")
    evaluation.add_argument("--output", type=Path, default=ROOT / "outputs")
    retrieval = commands.add_parser("search")
    retrieval.add_argument("--query", type=Path, required=True)
    retrieval.add_argument("--directory", type=Path, required=True)
    retrieval.add_argument("--wavelet", choices=["haar", "db2", "db4"], default="haar")
    retrieval.add_argument("--top-k", type=int, default=5)
    retrieval.add_argument("--output", type=Path, default=ROOT / "outputs" / "search.csv")
    args = parser.parse_args()
    if args.command == "prepare-demo":
        prepare_demo(args.output)
    elif args.command == "evaluate":
        evaluate(args.manifest, args.output)
    else:
        search(args.query, args.directory, args.wavelet, args.top_k, args.output)


if __name__ == "__main__":
    main()
