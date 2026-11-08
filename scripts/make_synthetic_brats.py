"""Generate a small *synthetic* BraTS-2021-shaped dataset (no real data needed).

Used by the 3D tests and by the dry run: it writes one ``BraTS2021_*`` directory per case
with the four modalities, the segmentation, a proper affine (so the spacing is real mm) and
a couple of patients **without** enhancing tumour, so the empty-ET bookkeeping is exercised.

    python scripts/make_synthetic_brats.py --root brats2021_synthetic --cases 6

Nothing here is anatomical: it exists to smoke-test the pipeline (shapes, spacing,
normalization, crops, sliding-window inference, VRAM and throughput), never to report a
metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT := _REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

MODALITIES: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="brats2021_synthetic", help="output directory")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[160, 160, 160],
        metavar=("D", "H", "W"),
        help="volume shape (default 160^3: big enough for a 128^3 crop)",
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 2.0], metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--empty-et",
        type=int,
        default=1,
        help="how many patients get no enhancing tumour (empty reference)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not root.is_absolute():
        root = _REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    shape = tuple(int(value) for value in args.shape)
    spacing = tuple(float(value) for value in args.spacing)
    affine = np.diag([*spacing, 1.0])

    for index in range(int(args.cases)):
        case = f"BraTS2021_{index:05d}"
        directory = root / case
        directory.mkdir(parents=True, exist_ok=True)
        grid = np.indices(shape)
        centre = np.array([size / 2.0 for size in shape]) + rng.normal(0, 4.0, size=3)
        radius = np.array(shape) * rng.uniform(0.30, 0.4)
        # a smooth ellipsoidal "brain"
        distance = (((grid - centre.reshape(3, *([1] * 3))) / radius.reshape(3, *([1] * 3))) ** 2).sum(axis=0)
        brain = distance <= 1.0
        for modality_index, modality in enumerate(MODALITIES):
            values = (
                brain.astype(np.float32) * rng.uniform(80, 400)
                + rng.normal(0, 25, size=shape).astype(np.float32) * brain
                + modality_index * 30.0 * brain
            )
            image = nib.Nifti1Image(values.astype(np.int16), affine)
            nib.save(image, str(directory / f"{case}_{modality}.nii.gz"))
        labels = np.zeros(shape, dtype=np.int16)
        # whole tumour (label 1) inside the brain, ET (label 4) as a smaller ball in it
        tumour_centre = centre + rng.normal(0, 6.0, size=3)
        tumour = (
            ((grid - tumour_centre.reshape(3, *([1] * 3))) / (radius * 0.35).reshape(3, *([1] * 3))) ** 2
        ).sum(axis=0) <= 1.0
        labels[tumour] = 1
        if index >= int(args.empty_et):
            labels[tumour & (rng.random(shape) < 0.4)] = 4
        segmentation = nib.Nifti1Image(labels, affine)
        nib.save(segmentation, str(directory / f"{case}_seg.nii.gz"))
        et_voxels = int((labels == 4).sum())
        print(f"{case}: shape {shape}, spacing {spacing} mm, ET voxels {et_voxels}")
    print(f"\n{args.cases} synthetic case(s) under {root}")
    print("now run: python scripts/make_brats_splits.py --config configs/brats2021/segresnet.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
