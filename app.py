"""Interactive demo: upload a satellite image, pick features and an algorithm,
see the unsupervised segmentation.

    pip install gradio
    python app.py
"""

import numpy as np
import gradio as gr
from sklearn.cluster import KMeans

from sscluster import DESSC
from sscluster.features import build_features
from sscluster.segmentation import segment

PALETTE = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
], dtype=np.uint8)


def run(image, n_clusters, algorithm, feature_kinds, use_superpixels, overlay):
    if image is None:
        return None
    img = np.asarray(image)[..., :3]
    if max(img.shape[:2]) > 768:                      # keep the demo responsive
        from PIL import Image
        scale = 768 / max(img.shape[:2])
        img = np.asarray(Image.fromarray(img).resize(
            (int(img.shape[1] * scale), int(img.shape[0] * scale))))
    kinds = feature_kinds or ["thesis"]
    feats = build_features(img, kinds)
    model = (DESSC(int(n_clusters), max_iter=30, init="k-means++", random_state=0) if algorithm == "DESSC"
             else KMeans(int(n_clusters), n_init=10, random_state=0))
    labels = segment(img, model, features=feats,
                     superpixels=800 if use_superpixels else None)
    colour = PALETTE[labels % len(PALETTE)]
    if overlay:
        colour = (0.5 * img + 0.5 * colour).astype(np.uint8)
    return colour


with gr.Blocks(title="Satellite Subspace Clustering") as demo:
    gr.Markdown("# Satellite Subspace Clustering\nUnsupervised land-cover "
                "segmentation with DESSC (soft subspace clustering + differential evolution).")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Satellite image", type="numpy")
            k = gr.Slider(2, 8, value=4, step=1, label="Number of clusters")
            algo = gr.Radio(["DESSC", "k-means"], value="DESSC", label="Algorithm")
            kinds = gr.CheckboxGroup(["thesis", "lab", "hsv", "lbp", "gabor"],
                                     value=["thesis"], label="Features")
            sp = gr.Checkbox(value=True, label="Cluster superpixels (faster, cleaner)")
            ov = gr.Checkbox(value=False, label="Overlay on image")
            btn = gr.Button("Segment", variant="primary")
        out = gr.Image(label="Segmentation")
    btn.click(run, [inp, k, algo, kinds, sp, ov], out)

if __name__ == "__main__":
    demo.launch()
