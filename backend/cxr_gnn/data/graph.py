"""
data/graph.py — Image → PyG graph conversion (inference path).

Node features (140-d):
    [0:128]  Deep features: ResNet18 layer2 feature map, average-pooled per superpixel
    [128:140] Hand-crafted: intensity stats, LBP texture, shape descriptors

Edge features (2-d, optional):
    [0] intensity difference between adjacent regions
    [1] LBP texture difference

This must stay pixel-for-pixel identical to the training pipeline, or the
trained model will see out-of-distribution features at inference time.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from skimage.segmentation import slic
from torch_geometric.data import Data

from cxr_gnn.config import Config


def _rag_edges(labels: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()

    a, b = labels[:, :-1].ravel(), labels[:, 1:].ravel()
    m = a != b
    for u, v in zip(a[m], b[m]):
        edge_set.add((min(u, v), max(u, v)))

    a, b = labels[:-1, :].ravel(), labels[1:, :].ravel()
    m = a != b
    for u, v in zip(a[m], b[m]):
        edge_set.add((min(u, v), max(u, v)))

    if not edge_set:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.array(sorted(edge_set), dtype=np.int64)
    edges -= 1
    return edges


def _region_pool(
    fmap: np.ndarray,
    flat_labels: np.ndarray,
    pixel_counts: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    C = fmap.shape[0]
    sums = np.zeros((C, n_nodes), dtype=np.float64)
    for c in range(C):
        np.add.at(sums[c], flat_labels, fmap[c])
    return (sums / np.clip(pixel_counts, 1, None)).T.astype(np.float32)


def image_to_graph(
    img: np.ndarray,
    cfg: Config,
    encoder,
    device: torch.device,
) -> Optional[Data]:
    """Convert a grayscale float32 [0,1] (H,W) image to a PyG Data graph.

    Returns None if SLIC produces a degenerate graph (<3 superpixels or 0 edges) —
    the caller should treat this as "image not usable".
    """
    H, W = img.shape

    labels = slic(
        img,
        n_segments=cfg.n_segments,
        compactness=cfg.compactness,
        channel_axis=None,
        start_label=1,
    )
    n_nodes = int(labels.max())
    if n_nodes < 3:
        return None

    idx = np.arange(1, n_nodes + 1)

    img_u8 = (img * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_u8, cfg.lbp_p, cfg.lbp_r, method="uniform")

    mean_i = ndi.mean(img, labels, index=idx)
    std_i  = ndi.standard_deviation(img, labels, index=idx)
    min_i  = ndi.minimum(img, labels, index=idx)
    max_i  = ndi.maximum(img, labels, index=idx)
    lbp_m  = ndi.mean(lbp, labels, index=idx)
    lbp_s  = ndi.standard_deviation(lbp, labels, index=idx)

    props = regionprops(labels)
    cy  = np.array([p.centroid[0] for p in props]) / H
    cx  = np.array([p.centroid[1] for p in props]) / W
    area = np.array([p.area for p in props]) / float(H * W)
    ecc  = np.array([p.eccentricity for p in props])
    sol  = np.array([p.solidity for p in props])
    ext  = np.array([p.extent for p in props])

    hand = np.stack(
        [mean_i, std_i, min_i, max_i,
         lbp_m / 255.0, lbp_s / 255.0,
         cy, cx, area, ecc, sol, ext],
        axis=1,
    ).astype(np.float32)  # (N, 12)

    if encoder is not None:
        import torch.nn.functional as F

        MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            t = torch.from_numpy(img)[None, None].float().to(device).repeat(1, 3, 1, 1)
            t = (t - MEAN) / STD
            fmap = encoder(t)
            fmap = F.interpolate(fmap, size=(H, W), mode="bilinear", align_corners=False)
            fmap_np = fmap[0].cpu().numpy()

        flat_labels = labels.ravel() - 1
        pixel_counts = np.bincount(flat_labels, minlength=n_nodes).astype(np.float32)
        deep = _region_pool(fmap_np.reshape(cfg.deep_feat_dim, -1),
                             flat_labels, pixel_counts, n_nodes)

        x = np.concatenate([deep, hand], axis=1).astype(np.float32)  # (N, 140)
    else:
        x = hand

    e = _rag_edges(labels)
    if e.shape[0] == 0:
        return None

    edge_index = np.concatenate([e.T, e.T[::-1]], axis=1)

    if cfg.use_edge_feat:
        d_int = np.abs(mean_i[e[:, 0]] - mean_i[e[:, 1]])
        d_tex = np.abs(lbp_m[e[:, 0]] - lbp_m[e[:, 1]]) / 255.0
        edge_attr_np = np.stack([d_int, d_tex], axis=1).astype(np.float32)
        edge_attr_np = np.concatenate([edge_attr_np, edge_attr_np], axis=0)
        edge_attr = torch.tensor(edge_attr_np)
    else:
        edge_attr = None

    data = Data(
        x=torch.tensor(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=edge_attr,
    )
    data.labels = labels
    return data
