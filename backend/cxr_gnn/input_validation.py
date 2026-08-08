"""Fail-closed chest-X-ray input validation.

The disease model was trained only on chest radiographs. Its final softmax
always assigns a disease label, including to cats, screenshots, and ordinary
photos. This module is a separate gate that must approve an input before the
disease model is allowed to run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from cxr_gnn.data.dataset import inspect_and_load_image
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


# A standard ImageNet model reliably recognises the kinds of accidental inputs
# we must never send to a medical classifier (animals, clothing, phones, and
# screens). These classes get a lower threshold because screenshots often have
# a diffuse ImageNet distribution even when they are obviously not X-rays.
_DIGITAL_INPUT_CLASSES = frozenset({
    "cellular telephone", "computer keyboard", "desktop computer", "digital clock",
    "hand-held computer", "iPod", "laptop", "monitor", "notebook", "screen",
    "television", "web site",
})
_OBJECT_CONFIDENCE_THRESHOLD = 0.18
_DIGITAL_CONFIDENCE_THRESHOLD = 0.04


@dataclass(frozen=True)
class ValidationResult:
    """The validated grayscale image plus a response-safe decision."""

    image: np.ndarray
    accepted: bool
    reason: str | None = None
    detail: str | None = None
    diagnostics: dict[str, Any] | None = None


class ChestXrayInputValidator:
    """Independent pre-inference gate for chest radiographs.

    ``CXR_REQUIRE_DICOM=true`` is an optional deployment policy for settings
    that need an auditable source. It rejects every PNG/JPEG, including valid
    radiographs, and therefore provides the strongest source restriction.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._classifier: torch.nn.Module | None = None
        self._preprocess = None
        self._categories: list[str] = []
        self.require_dicom = os.environ.get("CXR_REQUIRE_DICOM", "false").lower() == "true"

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None and self._preprocess is not None

    def load(self) -> None:
        """Load the independent natural-image veto model once at startup.

        This intentionally has no permissive fallback. If the model weights
        are unavailable, prediction is withheld rather than running the
        disease classifier without an input validator.
        """
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            classifier = resnet18(weights=weights).to(self.device).eval()
            for parameter in classifier.parameters():
                parameter.requires_grad = False

            self._classifier = classifier
            self._preprocess = weights.transforms()
            self._categories = list(weights.meta["categories"])
            logger.info("Loaded independent ImageNet input-veto model.")
        except Exception:
            self._classifier = None
            self._preprocess = None
            logger.exception("Chest-X-ray input validator could not be loaded.")

    def validate(self, raw_bytes: bytes, size: int) -> ValidationResult:
        """Decode and validate an upload before disease inference.

        No probabilities from the disease classifier are produced until this
        function accepts the image.
        """
        image, checks = inspect_and_load_image(raw_bytes, size)

        if not checks["passed"]:
            reason = checks["reason"] or "non_chest_xray"
            return ValidationResult(
                image=image,
                accepted=False,
                reason=reason,
                detail=self._detail_for_basic_rejection(reason),
                diagnostics=checks,
            )

        if self.require_dicom and checks["source_type"] != "dicom":
            return ValidationResult(
                image=image,
                accepted=False,
                reason="untrusted_raster_source",
                detail=(
                    "This deployment accepts only DICOM chest-radiography files. "
                    "PNG/JPEG uploads are withheld because their imaging source "
                    "cannot be verified."
                ),
                diagnostics=checks,
            )

        if self.require_dicom:
            metadata = checks["dicom_metadata"]
            is_chest_radiograph = (
                metadata.get("modality") in {"CR", "DX"}
                and (
                    "CHEST" in metadata.get("body_part", "")
                    or "CHEST" in metadata.get("study_description", "")
                )
            )
            if not is_chest_radiograph:
                return ValidationResult(
                    image=image,
                    accepted=False,
                    reason="unverified_dicom_metadata",
                    detail=(
                        "Prediction withheld: the DICOM metadata does not identify "
                        "this upload as a chest CR/DX radiograph."
                    ),
                    diagnostics=checks,
                )

        if not self.is_ready:
            return ValidationResult(
                image=image,
                accepted=False,
                reason="validator_unavailable",
                detail=(
                    "Prediction withheld because the independent chest-X-ray "
                    "input validator is unavailable. Please try again later."
                ),
                diagnostics=checks,
            )

        # ImageNet is useful as a veto for obvious objects, but it is not a
        # chest-X-ray classifier. Give a structurally plausible radiograph
        # priority so a grayscale X-ray is not mislabeled as an ImageNet item
        # such as “iPod”, “fountain”, or “spotlight”.
        radiograph_likeness = self._radiograph_likeness(image, checks)
        checks = {**checks, "radiograph_likeness": radiograph_likeness}
        if radiograph_likeness:
            return ValidationResult(image=image, accepted=True, diagnostics=checks)

        label, confidence = self._natural_image_prediction(image)
        checks = {
            **checks,
            "natural_image_label": label,
            "natural_image_confidence": confidence,
        }
        is_digital_input = label in _DIGITAL_INPUT_CLASSES and confidence >= _DIGITAL_CONFIDENCE_THRESHOLD
        is_recognised_object = confidence >= _OBJECT_CONFIDENCE_THRESHOLD
        if is_digital_input or is_recognised_object:
            return ValidationResult(
                image=image,
                accepted=False,
                reason="recognised_non_medical_image",
                detail=(
                    "This is a non-medical image, not a chest X-ray."
                ),
                diagnostics=checks,
            )

        return ValidationResult(image=image, accepted=True, diagnostics=checks)

    @staticmethod
    def _radiograph_likeness(image: np.ndarray, checks: dict[str, Any]) -> bool:
        """Conservative frontal-radiograph structure check.

        This is intentionally only an allow-list signal. It requires a nearly
        monochrome image, a portrait-like radiograph aspect ratio, substantial
        left/right symmetry, and a brighter central mediastinal band than the
        lateral fields. Ordinary colour photos and UI screenshots do not meet
        all four conditions, while the project's frontal chest X-rays do.
        """
        if checks.get("source_type") == "dicom":
            metadata = checks.get("dicom_metadata", {})
            if metadata.get("modality") in {"CR", "DX"}:
                body = f"{metadata.get('body_part', '')} {metadata.get('study_description', '')}"
                if "CHEST" in body:
                    return True

        if checks.get("colored_pixel_fraction", 1.0) > 0.05:
            return False
        if not (0.55 <= float(checks.get("aspect_ratio", 0.0)) <= 1.35):
            return False

        height, width = image.shape[:2]
        if height < 32 or width < 32:
            return False
        left = image[:, : width // 2]
        right = np.fliplr(image[:, width - width // 2 :])
        correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
        if not np.isfinite(correlation) or correlation < 0.66:
            return False

        center_start = int(width * 0.38)
        center_end = int(width * 0.62)
        side_width = max(1, int(width * 0.18))
        center_mean = float(image[:, center_start:center_end].mean())
        side_mean = float(
            np.concatenate((image[:, :side_width], image[:, -side_width:]), axis=1).mean()
        )
        return center_mean - side_mean >= 0.13

    @torch.no_grad()
    def _natural_image_prediction(self, image: np.ndarray) -> tuple[str, float]:
        assert self._classifier is not None
        assert self._preprocess is not None

        image_u8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_u8, mode="L").convert("RGB")
        batch = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self._classifier(batch), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        index_int = int(index.item())
        return self._categories[index_int], float(confidence.item())

    @staticmethod
    def _detail_for_basic_rejection(reason: str) -> str:
        details = {
            "multi_colored_photo": (
                "Prediction withheld: a coloured photo or screen image was detected. "
                "Upload a monochrome chest X-ray instead."
            ),
            "non_xray_histogram": (
                "Prediction withheld: the image tone profile does not match a chest X-ray."
            ),
            "synthetic_edge_pattern": (
                "Prediction withheld: the upload resembles a document, diagram, or vector graphic."
            ),
            "invalid_aspect_ratio": (
                "Prediction withheld: the image dimensions are not suitable for chest X-ray analysis."
            ),
        }
        return details.get(reason, "Prediction withheld: this upload could not be verified as a chest X-ray.")
