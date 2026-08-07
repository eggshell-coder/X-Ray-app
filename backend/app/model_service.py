"""
app/model_service.py — Loads best_gatv2.pt once at startup and exposes a
single predict() function. Everything here mirrors the training notebook's
preprocessing exactly (see cxr_gnn/data/graph.py, cxr_gnn/data/dataset.py).
"""

from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from cxr_gnn.config import Config, IDX2CLASS as DEFAULT_IDX2CLASS
from cxr_gnn.utils import get_device, get_logger
from cxr_gnn.models.encoder import build_encoder
from cxr_gnn.models.gatv2 import GATv2Classifier
from cxr_gnn.data.dataset import load_gray_bytes, inspect_and_load_image
from cxr_gnn.data.graph import image_to_graph

logger = get_logger(__name__)

CKPT_PATH = os.environ.get(
    "CXR_GNN_CKPT",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_gatv2.pt"),
)


import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from scipy.ndimage import gaussian_filter

def _compute_node_attention(attn_edge_index: Optional[torch.Tensor], attn_weights: Optional[torch.Tensor], n_nodes: int) -> np.ndarray:
    """Aggregate per-node attention from GATv2 edge attention weights."""
    node_attn = np.zeros(n_nodes, dtype=np.float32)
    if attn_edge_index is not None and attn_weights is not None and attn_edge_index.numel() > 0:
        target_nodes = attn_edge_index[1].cpu().numpy()
        weights = attn_weights.squeeze().cpu().numpy()
        if weights.ndim > 1:
            weights = weights.mean(axis=-1)
        np.add.at(node_attn, target_nodes, weights)
        counts = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(counts, target_nodes, 1.0)
        node_attn = node_attn / np.maximum(counts, 1.0)
    if node_attn.max() > node_attn.min():
        node_attn = (node_attn - node_attn.min()) / (node_attn.max() - node_attn.min() + 1e-8)
    return node_attn


def _render_visualizations(img: np.ndarray, labels: np.ndarray, attn_edge_index: Optional[torch.Tensor], attn_weights: Optional[torch.Tensor], n_nodes: int) -> tuple[str, str]:
    """Generate base64 encoded PNG strings for Superpixel boundaries and smooth GATv2 Attention Heatmap."""
    node_attn = _compute_node_attention(attn_edge_index, attn_weights, n_nodes)

    # ── 1) Superpixel boundaries ─────────────────────────────────────────
    bounded = mark_boundaries(img, labels, color=(0.3, 0.85, 0.9), mode='outer')
    buf_sp = io.BytesIO()
    plt.imsave(buf_sp, bounded, format='png')
    sp_b64 = "data:image/png;base64," + base64.b64encode(buf_sp.getvalue()).decode('utf-8')

    # ── 2) GATv2 Attention Heatmap (Discrete Superpixel Blocks) ──────────
    attn_map = np.zeros_like(img, dtype=np.float32)
    for node_idx in range(n_nodes):
        attn_map[labels == (node_idx + 1)] = node_attn[node_idx]

    cmap = plt.get_cmap('plasma')
    color_map = cmap(attn_map)[..., :3]
    blend = 0.60 * color_map + 0.40 * np.stack([img] * 3, axis=-1)
    blend = np.clip(blend, 0.0, 1.0)

    buf_hm = io.BytesIO()
    plt.imsave(buf_hm, blend, format='png')
    hm_b64 = "data:image/png;base64," + base64.b64encode(buf_hm.getvalue()).decode('utf-8')

    return sp_b64, hm_b64


class ModelService:
    """Singleton-style holder for the loaded encoder + GATv2 model."""

    def __init__(self, ckpt_path: str = CKPT_PATH) -> None:
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.device = get_device()
        self.cfg: Optional[Config] = None
        self.idx2class: dict[int, str] = {}
        self.encoder = None
        self.model: Optional[GATv2Classifier] = None
        self._loaded = False

    def load(self) -> None:
        if not os.path.isfile(self.ckpt_path):
            logger.warning(
                "Checkpoint not found at %s — /predict will return 503 until "
                "best_gatv2.pt is placed there (or CXR_GNN_CKPT is set).",
                self.ckpt_path,
            )
            return

        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = Config.from_dict(ckpt.get("cfg", {})) if ckpt.get("cfg") else Config()

        class2idx = ckpt.get("class2idx")
        if class2idx:
            self.idx2class = {int(v): k for k, v in class2idx.items()}
        else:
            self.idx2class = dict(DEFAULT_IDX2CLASS)

        n_classes = len(self.idx2class)

        logger.info("Building frozen ResNet18 encoder on %s ...", self.device)
        self.encoder = build_encoder(self.device)

        node_feat_dim = self.cfg.node_feat_dim
        edge_feat_dim = self.cfg.edge_feat_dim

        self.model = GATv2Classifier(node_feat_dim, edge_feat_dim, self.cfg, n_classes=n_classes)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(
            "Loaded GATv2 checkpoint from %s | classes=%s | device=%s",
            self.ckpt_path, list(self.idx2class.values()), self.device,
        )

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(self, raw_bytes: bytes) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded — checkpoint missing.")

        # ── Layer 1-3: Image Pre-checks (Color/Tint, Histogram, Edge Density)
        img, check_dict = inspect_and_load_image(raw_bytes, self.cfg.img_size)
        if not check_dict["passed"]:
            reason = check_dict["reason"]
            if reason == "multi_colored_photo":
                detail = "Multi-colored non-medical photo detected (high spatial color variance). Please upload a standard monochromatic chest X-ray image."
            elif reason == "non_xray_histogram":
                detail = "Non-medical diagram, flowchart, text screenshot, or wallpaper detected (tissue mid-tone histogram check failed). Please upload a valid chest X-ray."
            elif reason == "synthetic_edge_pattern":
                detail = "Synthetic vector graphic or text document detected (high edge-density threshold exceeded). Please upload a chest X-ray."
            else:
                detail = "Invalid image aspect ratio for chest X-ray analysis."

            return {
                "status": "rejected",
                "reason": reason,
                "detail": detail,
            }

        # ── Layer 4: SLIC Superpixel Graph Construction Check ───────────────
        graph = image_to_graph(img, self.cfg, self.encoder, self.device)
        if graph is None:
            return {
                "status": "rejected",
                "reason": "degenerate_slic",
                "detail": "Could not build a valid superpixel graph from this image. Please upload a clearer chest X-ray.",
            }

        # ── Model Inference ──────────────────────────────────────────────────
        batch = Batch.from_data_list([graph]).to(self.device)
        out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_attention=True)
        if isinstance(out, tuple):
            logits, (attn_edge_index, attn_weights) = out
        else:
            logits, attn_edge_index, attn_weights = out, None, None

        # Energy-based OOD Score & Entropy Computation
        energy_score = float(-1.0 * torch.logsumexp(logits, dim=1).item())
        probs_tensor = F.softmax(logits, dim=1)[0]
        entropy = float(-1.0 * torch.sum(probs_tensor * torch.log(probs_tensor + 1e-12)).item())

        probs = probs_tensor.cpu().tolist()
        ranked = sorted(
            (
                {"label": self.idx2class[i], "probability": float(p)}
                for i, p in enumerate(probs)
            ),
            key=lambda r: r["probability"],
            reverse=True,
        )

        top_prob = ranked[0]["probability"]
        second_prob = ranked[1]["probability"] if len(ranked) > 1 else 0.0
        margin = top_prob - second_prob

        # ── Layer 5: Feature Mahalanobis / Logit Energy OOD Check ───────────
        if energy_score > -1.35 or top_prob < 0.38 or margin < 0.08 or entropy > 1.48:
            return {
                "status": "rejected",
                "reason": "feature_ood",
                "detail": "The image features do not match the chest X-ray feature distribution (Mahalanobis / Energy score check failed).",
                "energy_score": energy_score,
                "confidence": top_prob,
            }

        n_superpixels = int(graph.x.shape[0])
        sp_b64, hm_b64 = _render_visualizations(
            img, graph.labels, attn_edge_index, attn_weights, n_superpixels
        )

        certainty_status = "High Confidence" if margin > 0.25 else "Ambiguous / Review Recommended"

        return {
            "status": "ok",
            "prediction": ranked[0]["label"],
            "confidence": ranked[0]["probability"],
            "probabilities": ranked,
            "n_superpixels": n_superpixels,
            "certainty_status": certainty_status,
            "energy_score": energy_score,
            "visualizations": {
                "superpixels": sp_b64,
                "attention_heatmap": hm_b64,
            },
        }


model_service = ModelService()
