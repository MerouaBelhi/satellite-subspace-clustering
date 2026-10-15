"""Dense self-supervised features from DINOv2 for pixel clustering.

The ViT returns one token per 14x14 patch. Tokens are reduced with PCA
and bilinearly upsampled back to image resolution, giving an (H, W, k)
feature map that `build_features(..., extra=...)` can stack with
hand-crafted features.

Requires `torch` (and an internet connection the first time, to fetch the
weights through torch.hub). The post-processing in
`tokens_to_pixel_features` is plain NumPy so it can be tested without torch.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

PATCH = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    """Project rows of X onto their top principal components."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def tokens_to_pixel_features(tokens: np.ndarray, grid: tuple[int, int],
                             out_shape: tuple[int, int],
                             n_components: int | None = 32) -> np.ndarray:
    """(N_patches, d) tokens on a (gh, gw) grid -> (H, W, k) feature map."""
    gh, gw = grid
    if tokens.shape[0] != gh * gw:
        raise ValueError(f"{tokens.shape[0]} tokens do not fit a {gh}x{gw} grid")
    if n_components is not None and n_components < tokens.shape[1]:
        tokens = pca_reduce(tokens, n_components)
    fmap = tokens.reshape(gh, gw, -1)
    H, W = out_shape
    return zoom(fmap, (H / gh, W / gw, 1), order=1)


class DINOv2Extractor:
    """Wraps a DINOv2 backbone loaded through torch.hub.

    Parameters
    ----------
    model_name : str
        One of dinov2_vits14 (fastest), dinov2_vitb14, dinov2_vitl14.
    n_components : int | None
        PCA dimensions kept per pixel (None keeps all 384/768/1024).
    max_side : int
        The image is resized so its longer side is at most this many pixels
        (rounded to a multiple of 14) before entering the network.
    """

    def __init__(self, model_name: str = "dinov2_vits14", n_components: int | None = 32,
                 max_side: int = 518, device: str | None = None):
        import torch  # imported lazily so the rest of the package needs no torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True).to(self.device).eval()
        self.n_components = n_components
        self.max_side = max_side

    def __call__(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        H, W = img.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        h = max(PATCH, int(round(H * scale / PATCH)) * PATCH)
        w = max(PATCH, int(round(W * scale / PATCH)) * PATCH)
        x = np.asarray(Image.fromarray(img).resize((w, h), Image.BICUBIC), float) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = self.torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(self.device)
        with self.torch.no_grad():
            tokens = self.model.forward_features(t)["x_norm_patchtokens"][0]
        return tokens_to_pixel_features(tokens.cpu().numpy(), (h // PATCH, w // PATCH),
                                        (H, W), self.n_components)
