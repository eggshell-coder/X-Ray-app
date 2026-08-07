"""
config.py — Immutable configuration using frozen dataclass.

Kept field-for-field identical to the training notebook's config so that
Config(**checkpoint["cfg"]) reconstructs cleanly at inference time.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/datasets/shakib0hasan/capstone-c-dataset/capstone_avocado_version_three"
    img_size: int = 256
    valid_ext: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # ── Augmentation (unused at inference) ──────────────────────────────────
    do_aug: bool = True
    aug_cap: int = 220
    max_aug_per_img: int = 6

    # ── Superpixel graph ─────────────────────────────────────────────────────
    n_segments: int = 180
    compactness: float = 10.0
    lbp_p: int = 8
    lbp_r: float = 1.0

    # ── Feature flags ────────────────────────────────────────────────────────
    use_edge_feat: bool = True
    deep_feat_dim: int = 128
    hand_feat_dim: int = 12

    # ── Data split (unused at inference) ────────────────────────────────────
    test_size: float = 0.10
    val_size: float = 0.10

    # ── Model ────────────────────────────────────────────────────────────────
    hidden: int = 48
    heads: int = 4
    dropout: float = 0.40
    in_dropout: float = 0.15
    dropedge: float = 0.10
    label_smooth: float = 0.05

    # ── Training (unused at inference) ──────────────────────────────────────
    epochs: int = 150
    batch_size: int = 32
    lr: float = 3e-4
    wd: float = 3e-4
    patience: int = 6
    early_stop: int = 20
    grad_clip: float = 2.0

    # ── Cross-validation (unused at inference) ──────────────────────────────
    n_folds: int = 5
    cv_epochs: int = 120
    cv_patience: int = 18

    # ── Conformal prediction (unused at inference) ──────────────────────────
    conformal_alpha: float = 0.10
    raps_k_reg: int = 1
    raps_lam: float = 0.10

    # ── Statistical robustness suite (unused at inference) ──────────────────
    n_bootstrap: int = 2000
    sensitivity_seeds: tuple = (42, 43, 44, 45, 46)
    sensitivity_epochs: int = 100
    sensitivity_patience: int = 15
    include_cnn_baseline: bool = True

    # ── I/O ──────────────────────────────────────────────────────────────────
    work_dir: str = "/kaggle/working"
    cache_file: str = "/kaggle/working/graph_cache.pt"
    ckpt_file: str = "/kaggle/working/best_gatv2.pt"
    rebuild_cache: bool = False

    seed: int = 42

    @property
    def node_feat_dim(self) -> int:
        return self.deep_feat_dim + self.hand_feat_dim  # 140

    @property
    def edge_feat_dim(self) -> int | None:
        return 2 if self.use_edge_feat else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["valid_ext"] = list(self.valid_ext)
        d["sensitivity_seeds"] = list(self.sensitivity_seeds)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Config":
        """Reconstruct from a checkpoint's cfg dict, tolerating extra/missing keys."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(Config)}
        clean = {k: v for k, v in d.items() if k in valid_fields}
        if "valid_ext" in clean:
            clean["valid_ext"] = tuple(clean["valid_ext"])
        if "sensitivity_seeds" in clean:
            clean["sensitivity_seeds"] = tuple(clean["sensitivity_seeds"])
        return Config(**clean)


# ── Label mapping ────────────────────────────────────────────────────────────
# Fallback class list — the checkpoint's own "class2idx" is preferred at
# inference time (see app/model_service.py), this is only a default.
RAW2CLEAN: dict[str, str] = {
    "Cardiac Pathology":    "Cardiac",
    "Cronic Lung Disease":  "ChronicLung",
    "Chronic Lung Disease": "ChronicLung",
    "Normal":               "Normal",
    "TB":                   "TB",
    "Tuberculosis":         "TB",
    "plural Pathology":     "Pleural",
    "pleural Pathology":    "Pleural",
    "Pleural Pathology":    "Pleural",
}

CLEAN_LABELS: list[str] = sorted(set(RAW2CLEAN.values()))
CLASS2IDX: dict[str, int] = {c: i for i, c in enumerate(CLEAN_LABELS)}
IDX2CLASS: dict[int, str] = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES: int = len(CLEAN_LABELS)
