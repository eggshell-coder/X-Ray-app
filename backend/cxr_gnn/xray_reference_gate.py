"""No-training chest X-ray input gate based on a reference image bank.

Use this *before* the deployed disease classifier.  It has two allow paths:

1. A perceptual-hash match for an image that is already in the approved
   training dataset (including a resize/re-encode of that image).
2. A structure-similarity match against the approved chest X-ray reference
   bank for a new but visually similar chest radiograph.

It deliberately does not reject images because they are coloured, square,
low-edge, asymmetric, or labelled "person" by an ImageNet model.  Those
rules reject real chest X-rays in this project dataset.

The default similarity threshold was calibrated on the 570 images in this
repository using leave-one-out nearest-reference scoring: 564/570 (99.0%)
pass at 0.7287.  Keep the same descriptor/preprocessing when using it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, UnidentifiedImageError


ImageSource = Union[str, Path, BinaryIO]

DEFAULT_SIMILARITY_THRESHOLD = 0.7287
DEFAULT_PHASH_MAX_DISTANCE = 6
MIN_IMAGE_SIDE = 128
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class GateDecision:
    """The result returned to the web/API layer."""

    accepted: bool
    reason: str
    similarity: float | None
    nearest_reference: str | None
    phash_distance: int | None
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _open_grayscale(source: ImageSource) -> tuple[Image.Image, tuple[int, int]]:
    """Decode an image once and return its grayscale pixels and original size."""

    try:
        with Image.open(source) as opened:
            opened.load()  # fail here for truncated/corrupt uploads
            original_size = opened.size
            return opened.convert("L"), original_size
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The upload is not a readable image file.") from error


def _normalised_pixels(image: Image.Image) -> np.ndarray:
    """Crop only a small scanner border and normalise exposure per image."""

    width, height = image.size
    border_x, border_y = int(width * 0.04), int(height * 0.04)
    if width - 2 * border_x < 2 or height - 2 * border_y < 2:
        raise ValueError("The upload is too small to analyse.")

    image = image.crop((border_x, border_y, width - border_x, height - border_y))
    pixels = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(pixels, (2, 98))
    return np.clip((pixels - low) / max(high - low, 1.0), 0.0, 1.0)


def _unit_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values / (np.linalg.norm(values) + 1e-6)


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix.astype(np.float32)


_DCT_32 = _dct_matrix(32)


def _perceptual_hash(pixels: np.ndarray) -> np.uint64:
    """Return a standard 64-bit DCT perceptual hash without extra packages."""

    tiny = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    tiny = tiny.resize((32, 32), Image.Resampling.BILINEAR)
    values = np.asarray(tiny, dtype=np.float32)
    coefficients = _DCT_32 @ values @ _DCT_32.T
    low_frequency = coefficients[:8, :8].ravel()
    median = np.median(low_frequency[1:])  # DC component must not set the threshold
    bits = low_frequency > median
    bits[0] = False

    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return np.uint64(result)


def _descriptor(source: ImageSource) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.uint64]:
    """Create the exact descriptor used for reference-bank calibration."""

    grayscale, (width, height) = _open_grayscale(source)
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"The image is smaller than {MIN_IMAGE_SIDE}px on one side.")

    pixels = _normalised_pixels(grayscale)

    # A fixed resolution is used for structural comparison.  Aspect ratio is
    # retained separately as a deliberately low-weight feature rather than a
    # hard portrait-only rule.
    structure_image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    structure_image = structure_image.resize((80, 80), Image.Resampling.BILINEAR)
    structure = np.asarray(structure_image, dtype=np.float32) / 255.0

    structure_vector = structure.ravel()
    structure_vector = (structure_vector - structure_vector.mean()) / (structure_vector.std() + 1e-6)
    structure_vector = _unit_vector(structure_vector)

    histogram = np.histogram(pixels, bins=32, range=(0.0, 1.0))[0].astype(np.float32)
    histogram = _unit_vector(histogram)

    profile = np.concatenate((structure.mean(axis=0), structure.mean(axis=1)))
    profile = (profile - profile.mean()) / (profile.std() + 1e-6)
    profile = _unit_vector(profile)

    return structure_vector, histogram, profile, width / height, _perceptual_hash(pixels)


def build_reference_bank(dataset_dir: Union[str, Path], output_file: Union[str, Path]) -> int:
    """Build the one-time reference file from approved dataset images.

    This is feature precomputation, not model training.  The output contains
    no disease labels, because this gate only decides whether an image looks
    sufficiently similar to an approved chest radiograph.
    """

    root = Path(dataset_dir).resolve()
    output = Path(output_file).resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) < 2:
        raise ValueError("At least two approved X-ray images are required.")

    structures: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    aspects: list[float] = []
    hashes: list[np.uint64] = []
    paths: list[str] = []

    for image_file in files:
        structure, histogram, profile, aspect, image_hash = _descriptor(image_file)
        structures.append(structure)
        histograms.append(histogram)
        profiles.append(profile)
        aspects.append(aspect)
        hashes.append(image_hash)
        paths.append(str(image_file.relative_to(root)))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        structure=np.stack(structures).astype(np.float32),
        histogram=np.stack(histograms).astype(np.float32),
        profile=np.stack(profiles).astype(np.float32),
        aspect=np.asarray(aspects, dtype=np.float32),
        phash=np.asarray(hashes, dtype=np.uint64),
        path=np.asarray(paths, dtype="U512"),
    )
    return len(paths)


class ChestXrayReferenceGate:
    """A loaded reference bank suitable for re-use by a deployed app."""

    def __init__(
        self,
        reference_file: Union[str, Path],
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        phash_max_distance: int = DEFAULT_PHASH_MAX_DISTANCE,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        if phash_max_distance < 0:
            raise ValueError("phash_max_distance cannot be negative.")

        with np.load(Path(reference_file), allow_pickle=False) as bank:
            self.structure = bank["structure"].astype(np.float32)
            self.histogram = bank["histogram"].astype(np.float32)
            self.profile = bank["profile"].astype(np.float32)
            self.aspect = bank["aspect"].astype(np.float32)
            self.phash = bank["phash"].astype(np.uint64)
            self.paths = bank["path"].astype(str)

        if len(self.paths) == 0:
            raise ValueError("Reference bank has no images.")
        self.similarity_threshold = float(similarity_threshold)
        self.phash_max_distance = int(phash_max_distance)

    def check(self, uploaded_image: ImageSource) -> GateDecision:
        """Return whether an upload may proceed to the disease classifier."""

        try:
            structure, histogram, profile, aspect, image_hash = _descriptor(uploaded_image)
        except ValueError as error:
            return GateDecision(
                accepted=False,
                reason=str(error),
                similarity=None,
                nearest_reference=None,
                phash_distance=None,
                threshold=self.similarity_threshold,
            )

        distances = np.fromiter(
            (int(int(stored_hash ^ image_hash).bit_count()) for stored_hash in self.phash),
            dtype=np.int16,
            count=len(self.phash),
        )
        best_hash_index = int(np.argmin(distances))
        best_hash_distance = int(distances[best_hash_index])

        # Known approved dataset image: do not let a heuristic reject it.
        if best_hash_distance <= self.phash_max_distance:
            return GateDecision(
                accepted=True,
                reason="approved_dataset_image",
                similarity=None,
                nearest_reference=str(self.paths[best_hash_index]),
                phash_distance=best_hash_distance,
                threshold=self.similarity_threshold,
            )

        scores = (
            0.65 * (self.structure @ structure)
            + 0.20 * (self.histogram @ histogram)
            + 0.10 * (self.profile @ profile)
            + 0.05 * np.exp(-2.0 * np.abs(np.log(self.aspect / aspect)))
        )
        nearest_index = int(np.argmax(scores))
        best_score = float(scores[nearest_index])
        accepted = best_score >= self.similarity_threshold

        return GateDecision(
            accepted=accepted,
            reason="xray_like_reference" if accepted else "not_similar_to_reference_chest_xray",
            similarity=round(best_score, 4),
            nearest_reference=str(self.paths[nearest_index]),
            phash_distance=best_hash_distance,
            threshold=self.similarity_threshold,
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or test the no-training CXR reference gate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a reference bank from approved dataset images.")
    build.add_argument("dataset_dir", type=Path)
    build.add_argument("output_file", type=Path)

    check = subparsers.add_parser("check", help="Check one uploaded image against a reference bank.")
    check.add_argument("reference_file", type=Path)
    check.add_argument("image_file", type=Path)
    check.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)

    args = parser.parse_args()
    if args.command == "build":
        count = build_reference_bank(args.dataset_dir, args.output_file)
        print(f"Reference bank created with {count} approved X-ray images: {args.output_file}")
    else:
        decision = ChestXrayReferenceGate(args.reference_file, args.threshold).check(args.image_file)
        print(decision.to_dict())


if __name__ == "__main__":
    _main()
