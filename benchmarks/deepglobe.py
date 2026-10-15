"""Benchmark DESSC (paper init), DESSC++ (k-means++ init) and k-means on DeepGlobe.

Usage examples:
    python benchmarks/deepglobe.py DATA                          # thesis baseline
    python benchmarks/deepglobe.py DATA --features lab,lbp,gabor
    python benchmarks/deepglobe.py DATA --features thesis --superpixels 800
    python benchmarks/deepglobe.py DATA --features dinov2 --superpixels 800

Over-clustering (recommended for DeepGlobe, where one class often covers
most of the image):
    python benchmarks/deepglobe.py DATA --features thesis,lab,lbp --superpixels 800 --k 8

DATA holds DeepGlobe pairs <id>_sat.jpg / <id>_mask.png. The number of
clusters for each image is the number of classes present in its mask, as in
the thesis. Per-image scores are appended to results.csv.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sscluster import DESSC  # noqa: E402
from sscluster.features import build_features  # noqa: E402
from sscluster.segmentation import decode_deepglobe_mask, evaluate, segment  # noqa: E402


def load_pair(sat_path: Path, size: int):
    img = Image.open(sat_path).convert("RGB")
    mask_path = sat_path.with_name(sat_path.name.replace("_sat.jpg", "_mask.png"))
    mask = Image.open(mask_path).convert("RGB")
    if max(img.size) > size:   # DeepGlobe tiles are 2448 px; downscale for speed
        img = img.resize((size, size), Image.BILINEAR)
        mask = mask.resize((size, size), Image.NEAREST)
    return np.asarray(img), decode_deepglobe_mask(np.asarray(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--features", default="thesis",
                    help="comma list from thesis,lab,hsv,lbp,gabor,dinov2")
    ap.add_argument("--superpixels", type=int, default=0,
                    help="number of SLIC superpixels (0 = pixel sampling)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--k", type=int, default=0,
                    help="number of clusters (0 = number of classes in each mask)")
    ap.add_argument("--matching", choices=["hungarian", "majority"], default=None,
                    help="cluster-to-class matching; default hungarian if k=0, else majority")
    ap.add_argument("--shuffle", action="store_true",
                    help="take a random (seed 0) rather than alphabetical subset of images")
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    args = ap.parse_args()

    files = sorted(args.folder.glob("*_sat.jpg"))
    if args.shuffle:
        files = [files[i] for i in np.random.default_rng(0).permutation(len(files))]
    files = files[: args.limit]
    if not files:
        sys.exit(f"No *_sat.jpg files in {args.folder}")

    kinds = [k for k in args.features.split(",") if k]
    dino = None
    if "dinov2" in kinds:
        from sscluster.deep import DINOv2Extractor
        dino = DINOv2Extractor()
        kinds.remove("dinov2")
    matching = args.matching or ("hungarian" if args.k == 0 else "majority")
    config = f"{args.features}|sp={args.superpixels}|k={args.k or 'C'}|{matching}"

    UNKNOWN = 6   # DeepGlobe's official metric ignores the "unknown" class
    results = {"baseline": [], "DESSC": [], "DESSC++": [], "k-means": []}
    rows = []
    for f in files:
        img, gt = load_pair(f, args.size)
        C = len(np.setdiff1d(np.unique(gt), [UNKNOWN]))
        if C < 2:
            continue
        # Reference: label the whole image with its most frequent class
        base = evaluate(np.zeros_like(gt), gt, matching="majority", ignore=UNKNOWN)
        base["seconds"] = 0.0
        results["baseline"].append(base)
        rows.append({"config": config, "image": f.name, "algo": "baseline", "C": C, "K": 1,
                     "pixel_acc": base["pixel_acc"], "miou": base["miou"], "seconds": 0.0})
        feats = build_features(img, kinds, extra=dino(img) if dino else None)
        K = args.k or C
        models = {"DESSC": DESSC(K, random_state=0),
                  "DESSC++": DESSC(K, init="k-means++", random_state=0),
                  "k-means": KMeans(K, n_init=10, random_state=0)}
        for name, model in models.items():
            t0 = time.perf_counter()
            pred = segment(img, model, features=feats, superpixels=args.superpixels or None)
            r = evaluate(pred, gt, matching=matching, ignore=UNKNOWN)
            r["seconds"] = time.perf_counter() - t0
            results[name].append(r)
            rows.append({"config": config, "image": f.name, "algo": name, "C": C, "K": K,
                         "pixel_acc": r["pixel_acc"], "miou": r["miou"],
                         "seconds": r["seconds"]})
        print(f"{f.name}: C={C} K={K}  " + "  ".join(
            f"{n} mIoU={rs[-1]['miou']:.3f}" for n, rs in results.items()))

    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)

    print(f"\n[{config}] mean over {len(results['DESSC'])} images "
          "(baseline = whole image labelled with its majority class):")
    for name, rs in results.items():
        print(f"  {name:8} pixel acc={np.mean([r['pixel_acc'] for r in rs]):.4f}"
              f"  mIoU={np.mean([r['miou'] for r in rs]):.4f}"
              f"  time/img={np.mean([r['seconds'] for r in rs]):.1f}s")


if __name__ == "__main__":
    main()
