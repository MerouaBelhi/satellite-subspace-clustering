"""Generate synthetic satellite-like scenes with DeepGlobe-format masks.

Each scene is a Voronoi partition of land-cover regions, each rendered
with its own colour and texture (field stripes, tree canopy, building
blocks, ...). The output pairs <id>_sat.jpg / <id>_mask.png can be fed
directly to benchmarks/deepglobe.py or uploaded to app.py.

    python benchmarks/make_synthetic.py out_folder --n 5 --size 512
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# class index -> DeepGlobe mask colour
MASK_COLOURS = {0: (0, 255, 255), 1: (255, 255, 0), 2: (255, 0, 255),
                3: (0, 255, 0), 4: (0, 0, 255), 5: (255, 255, 255)}
URBAN, AGRI, RANGE, FOREST, WATER, BARREN = range(6)


def noise(shape, rng, sigma, amp):
    n = gaussian_filter(rng.normal(size=shape), sigma)
    return amp * n / (n.std() + 1e-9)


def render_region(cls, shape, rng):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    jitter = rng.uniform(-12, 12, 3)
    if cls == AGRI:
        base = np.array([150, 160, 80]) + rng.uniform(-30, 30, 3)
        angle = rng.uniform(0, np.pi)
        period = rng.uniform(6, 14)
        stripes = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 2 * np.pi / period)
        tex = 18 * stripes + noise(shape, rng, 1, 6)
    elif cls == FOREST:
        base = np.array([45, 75, 40])
        tex = noise(shape, rng, 1.5, 22)
    elif cls == WATER:
        base = np.array([25, 45, 70])
        tex = noise(shape, rng, 8, 4)
    elif cls == URBAN:
        base = np.array([125, 120, 115])
        block = rng.integers(10, 18)
        cells = ((yy // block) + (xx // block)) % 3
        roofs = np.where((yy % block > 2) & (xx % block > 2), cells * 25 - 25, -45)
        tex = roofs + noise(shape, rng, 0.8, 8)
    elif cls == BARREN:
        base = np.array([200, 180, 145])
        tex = noise(shape, rng, 0.7, 10)
    else:  # rangeland
        base = np.array([140, 125, 75])
        tex = noise(shape, rng, 3, 14) + noise(shape, rng, 0.7, 6)
    return np.clip(base + jitter + tex[..., None], 0, 255)


def make_scene(size, rng, n_regions=10, classes=(URBAN, AGRI, RANGE, FOREST, WATER, BARREN)):
    # Voronoi partition with wobbly borders (seeds measured on a warped grid)
    yy, xx = np.mgrid[:size, :size].astype(float)
    warp_y = yy + noise((size, size), rng, 25, 12)
    warp_x = xx + noise((size, size), rng, 25, 12)
    seeds = rng.uniform(0, size, (n_regions, 2))
    d = (warp_y[..., None] - seeds[:, 0]) ** 2 + (warp_x[..., None] - seeds[:, 1]) ** 2
    region = d.argmin(-1)
    # make sure at least 3 classes appear
    region_cls = rng.choice(classes, n_regions)
    region_cls[:3] = rng.choice(classes, 3, replace=False)
    gt = region_cls[region]

    img = np.zeros((size, size, 3))
    for r in range(n_regions):
        m = region == r
        img[m] = render_region(region_cls[r], (size, size), rng)[m]
    img = gaussian_filter(img, (0.7, 0.7, 0))     # sensor blur
    mask = np.zeros((size, size, 3), np.uint8)
    for c, col in MASK_COLOURS.items():
        mask[gt == c] = col
    return img.astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        img, mask = make_scene(args.size, rng)
        Image.fromarray(img).save(args.out / f"synth{i:02d}_sat.jpg", quality=92)
        Image.fromarray(mask).save(args.out / f"synth{i:02d}_mask.png")
    print(f"Wrote {args.n} scenes to {args.out}")


if __name__ == "__main__":
    main()
