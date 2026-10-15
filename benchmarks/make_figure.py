"""Build the README figure from real DeepGlobe tiles.

For each tile: input | ground truth | DESSC++ on the left features |
DESSC++ on the right features. Clusters are painted with the DeepGlobe
colour of the class they are matched to (majority matching, K clusters),
and each panel shows its mIoU ("unknown" ignored, as in the benchmark).

    python benchmarks/make_figure.py data/train 849797 882451
    python benchmarks/make_figure.py data/train 849797 --right lab,lbp   # without torch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepglobe import load_pair  # noqa: E402
from sscluster import DESSC  # noqa: E402
from sscluster.features import build_features  # noqa: E402
from sscluster.segmentation import evaluate, majority_match, segment  # noqa: E402

COLOURS = np.array([[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0],
                    [0, 0, 255], [255, 255, 255], [0, 0, 0]], np.uint8)
UNKNOWN = 6
LABELS = {"thesis": "Thesis features (colour)", "lab": "Lab", "lbp": "LBP",
          "gabor": "Gabor", "hsv": "HSV", "dinov2": "DINOv2 features"}


def describe(kinds):
    return " + ".join(LABELS.get(k, k) for k in kinds)


def features_for(img, kinds, dino):
    hand = [k for k in kinds if k != "dinov2"]
    extra = dino(img) if "dinov2" in kinds else None
    return build_features(img, hand, extra=extra)


def run(img, gt, kinds, dino, k, superpixels):
    feats = features_for(img, kinds, dino)
    pred = segment(img, DESSC(k, init="k-means++", random_state=0),
                   features=feats, superpixels=superpixels)
    score = evaluate(pred, gt, matching="majority", ignore=UNKNOWN)["miou"]
    painted = COLOURS[majority_match(pred, gt)]
    painted[gt == UNKNOWN] = COLOURS[UNKNOWN]
    return painted, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("ids", nargs="+", help="tile ids, e.g. 849797")
    ap.add_argument("--left", default="thesis")
    ap.add_argument("--right", default="dinov2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--superpixels", type=int, default=800)
    ap.add_argument("--panel", type=int, default=256, help="panel size in pixels")
    ap.add_argument("--out", type=Path, default=Path("docs/deepglobe_example.png"))
    args = ap.parse_args()

    left, right = args.left.split(","), args.right.split(",")
    dino = None
    if "dinov2" in left + right:
        from sscluster.deep import DINOv2Extractor
        dino = DINOv2Extractor()

    P, gap, head = args.panel, 8, 26
    titles = ["Satellite image", "Ground truth", describe(left), describe(right)]
    canvas = Image.new("RGB", (4 * P + 3 * gap, len(args.ids) * (P + head + gap)), "white")
    draw = ImageDraw.Draw(canvas)

    for row, tile in enumerate(args.ids):
        img, gt = load_pair(args.folder / f"{tile}_sat.jpg", 512)
        a, sa = run(img, gt, left, dino, args.k, args.superpixels)
        b, sb = run(img, gt, right, dino, args.k, args.superpixels)
        panels = [img, COLOURS[gt], a, b]
        captions = titles[:2] + [f"{titles[2]}  mIoU {sa:.2f}", f"{titles[3]}  mIoU {sb:.2f}"]
        y = row * (P + head + gap)
        for col, (p, cap) in enumerate(zip(panels, captions)):
            x = col * (P + gap)
            canvas.paste(Image.fromarray(p).resize((P, P), Image.NEAREST), (x, y + head))
            draw.text((x + 2, y + 7), cap, fill="black")
        print(f"{tile}: {describe(left)} mIoU={sa:.3f}   {describe(right)} mIoU={sb:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
