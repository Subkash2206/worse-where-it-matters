"""BraTS 2021: volumes, normalization, spacing, crops, folds, 3D TTA views, model, inferer.

Everything runs on small *synthetic* volumes written to ``tmp_path``; no BraTS data and no
GPU is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import load_config  # noqa: E402
from melba.data.brats import (  # noqa: E402
    BRATSDataset,
    BraTSAugmentation,
    ET_LABEL,
    MODALITIES,
    discover_brats_cases,
    foreground_crop,
    load_brats_items,
    percentile_normalize,
    read_brats_case,
)
from melba.data.splits import folded_splits, read_split_csv, split_file_for_fold, write_split_csv  # noqa: E402
from melba.inference.sliding import build_sliding_window, sliding_window_predict  # noqa: E402
from melba.transforms import build_views  # noqa: E402

CONFIG = _REPO_ROOT / "configs" / "brats2021" / "segresnet.yaml"
SHAPE = (16, 20, 24)  # (D, H, W): a tiny volume, real geometry
SPACING = (0.9, 1.1, 2.5)


def _write_case(root: Path, case: str, *, et: bool = True, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*SPACING, 1.0])
    brain = np.zeros(SHAPE, dtype=bool)
    brain[2:14, 3:17, 4:20] = True
    for index, modality in enumerate(MODALITIES):
        values = np.where(brain, rng.uniform(50, 200) + index * 20.0, 0.0)
        values = values + np.where(brain, rng.normal(0, 5, size=SHAPE), 0.0)
        nib.save(
            nib.Nifti1Image(values.astype(np.float32), affine),
            str(directory / f"{case}_{modality}.nii.gz"),
        )
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[6:10, 8:12, 9:13] = 1
    if et:
        labels[7:9, 9:11, 10:12] = ET_LABEL
    nib.save(
        nib.Nifti1Image(labels, affine), str(directory / f"{case}_seg.nii.gz")
    )


@pytest.fixture()
def brats_root(tmp_path: Path) -> Path:
    root = tmp_path / "brats2021"
    for index in range(5):
        _write_case(root, f"BraTS2021_{index:05d}", et=index != 3, seed=index)
    return root


def _config(tmp_path: Path, root: Path):
    return load_config(
        CONFIG,
        {
            "dataset.root": str(root),
            "dataset.patch_size": [8, 8, 8],  # tiny toy volumes: keep the crop smaller
            "dataset.use_cache": False,
            "dataset.split.output_pattern": str(tmp_path / "splits" / "brats2021_fold{fold}.csv"),
            "device": "cpu",
            "runtime.num_workers": 0,
            "evaluation.fold": 0,
            "training.checkpoint_dir": str(tmp_path / "checkpoints"),
            "evaluation.output_dir": str(tmp_path / "results"),
            "model.checkpoint": None,
            "model.init_filters": 8,
        },
    )


def test_discover_and_read_a_case_with_spacing_in_mm(brats_root: Path) -> None:
    root_config = _config(brats_root.parent, brats_root)
    cases = discover_brats_cases(root_config)
    assert cases == [f"BraTS2021_{index:05d}" for index in range(5)]

    from melba.data.brats import BraTSItem

    bra_item = BraTSItem(
        case_id=cases[0],
        group_id=cases[0],
        split="test",
        directory=brats_root / cases[0],
    )
    image, mask, spacing = read_brats_case(bra_item)
    assert image.shape == (len(MODALITIES), *SHAPE)
    assert mask.shape == SHAPE
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert int(mask.sum()) == 8  # 2*2*2 enhancing voxels
    assert spacing == pytest.approx(SPACING)  # read from the affine, in mm
    # per modality, over brain voxels: mean 0, unit variance, background exactly 0
    expected_brain = np.zeros(SHAPE, dtype=bool)
    expected_brain[2:14, 3:17, 4:20] = True  # what _write_case painted
    for channel in range(image.shape[0]):
        assert np.all(image[channel][~expected_brain] == 0.0)
        inside = image[channel][expected_brain]
        assert inside.mean() == pytest.approx(0.0, abs=1e-4)
        assert inside.std() == pytest.approx(1.0, abs=1e-3)


def test_percentile_normalize_clips_outliers_and_ignores_air() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 100.0
    volume[1, 1, 1] = 1e6  # an outlier that must be clipped away
    brain = np.zeros((4, 4, 4), dtype=bool)
    brain[1:3, 1:3, 1:3] = True
    normalized = percentile_normalize(volume, brain)
    assert normalized.max() < 10.0  # the outlier cannot dominate
    assert np.all(normalized[~brain] == 0.0)
    flat = percentile_normalize(np.full((2, 2, 2), 5.0, dtype=np.float32), np.ones((2, 2, 2), bool))
    assert np.all(flat == 0.0)  # a constant volume has nothing to normalize


def test_foreground_crop_is_centred_on_tumour_and_pads_small_volumes() -> None:
    image = np.zeros((4, 20, 20, 20), dtype=np.float32)
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    image[:, 5:7, 5:7, 5:7] = 1.0
    mask[5:7, 5:7, 5:7] = 1.0
    rng = np.random.default_rng(0)
    for _ in range(5):
        cropped_image, cropped_mask = foreground_crop(
            image, mask, (8, 8, 8), rng, foreground_p=1.0
        )
        assert cropped_image.shape == (4, 8, 8, 8)
        assert cropped_mask.sum() > 0  # a foreground crop always contains tumour
    padded_image, padded_mask = foreground_crop(
        image[:, :4, :4, :4], mask[:4, :4, :4], (8, 8, 8), rng, foreground_p=0.0
    )
    assert padded_image.shape == (4, 8, 8, 8) and padded_mask.shape == (8, 8, 8)


def test_case_level_folds_test_k_and_val_next(tmp_path: Path, brats_root: Path) -> None:
    """5-fold case CV: test = fold k, validation = the next fold, no case in two splits."""
    config = _config(tmp_path, brats_root)
    cases = discover_brats_cases(config)
    folds = folded_splits(cases, {case: case for case in cases}, n_folds=5, seed=2024)
    for fold, records in folds.items():
        write_split_csv(records, split_file_for_fold(config, fold), overwrite=True)
        sets = {
            split: {record.case_id for record in records if record.split == split}
            for split in ("train", "val", "test")
        }
        assert len(sets["test"]) == 1  # 5 cases, 5 folds: one patient per test fold
        assert not sets["test"] & sets["val"] and not sets["train"] & sets["test"]
        assert sets["train"] | sets["val"] | sets["test"] == set(cases)
    test_order = [next(r.case_id for r in folds[k] if r.split == "test") for k in range(5)]
    val_order = [next(r.case_id for r in folds[k] if r.split == "val") for k in range(5)]
    assert val_order == test_order[1:] + test_order[:1]  # rotating validation fold


def test_dataset_train_crop_and_eval_volume(tmp_path: Path, brats_root: Path) -> None:
    config = _config(tmp_path, brats_root)
    cases = discover_brats_cases(config)
    folds = folded_splits(cases, {case: case for case in cases}, n_folds=5, seed=2024)
    for fold, records in folds.items():
        write_split_csv(records, split_file_for_fold(config, fold), overwrite=True)

    from melba.data import build_dataset

    train = build_dataset(config, "train", train=True, fold=0)
    sample = train[0]
    assert sample["image"].shape == (4, 8, 8, 8)  # patch_size from the config
    assert sample["mask"].shape == (1, 8, 8, 8)
    assert sample["mask"].dtype == torch.float32
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["spacing"] == pytest.approx(SPACING)
    # the same index gives a different crop in the next epoch (reseeded stream)
    first = train[0]["image"]
    train.set_epoch(1)
    assert not torch.equal(first, train[0]["image"])

    evaluation = build_dataset(config, "test", train=False, fold=0)
    full = evaluation[0]
    assert full["image"].shape == (4, *SHAPE)  # whole volume, no crop, no resize
    assert full["mask_native"].shape == (1, *SHAPE)
    assert full["spacing"] == pytest.approx(SPACING)



def test_augmentation_groups_keep_image_and_mask_aligned() -> None:
    from melba.data.brats import BraTSItem

    image = np.zeros((4, 6, 7, 8), dtype=np.float32)
    image[:, 1:3, 2:5, 3:6] = 1.0
    mask = np.zeros((6, 7, 8), dtype=np.float32)
    mask[1:3, 2:5, 3:6] = 1.0
    for group in ("flips_only", "geometric_3d"):
        dataset = BRATSDataset(
            [BraTSItem("c", "c", "train", Path("."))],
            train=True,
            patch_size=(6, 7, 8),
            augmentation=BraTSAugmentation(group=group, flip_p=1.0, rot90_p=1.0),
        )
        for _ in range(4):
            augmented_image, augmented_mask = dataset._augment(
                image, mask, np.random.default_rng(0)
            )
            assert augmented_image.shape == image.shape
            assert augmented_mask.shape == mask.shape
            # the mask marks exactly the voxels the image paints, in every view
            painted = augmented_image.sum(axis=0) > 0
            assert np.array_equal(painted, augmented_mask > 0)
            assert augmented_mask.sum() == mask.sum()


def test_3d_tta_views_have_exact_inverses() -> None:
    """1/2/4/8 views from the BraTS config: every view round-trips a volume exactly."""
    config = load_config(CONFIG)
    methods = config.get("tta.methods")
    assert [method["name"] for method in methods] == ["baseline", "tta2", "tta4", "tta8"]
    assert [len(method["views"]) for method in methods] == [1, 2, 4, 8]
    volume = torch.arange(4 * 5 * 6 * 7, dtype=torch.float32).reshape(1, 4, 5, 6, 7)
    transformed_first_elements = set()
    for method in methods:
        views = build_views(method["views"])
        assert len({view.transform.name for view in views}) == len(views)
        for view in views:
            transformed = view.apply(volume)
            assert transformed.shape == volume.shape
            restored = view.invert(transformed, tuple(volume.shape[1:]))
            assert torch.equal(restored, volume)  # exact, not approximate
            transformed_first_elements.add(tuple(transformed.flatten().tolist()))
    assert len(transformed_first_elements) == 8  # the tta8 group really has 8 volumes


def test_segresnet_forward_and_sliding_window_inference() -> None:
    config = load_config(
        CONFIG, {"model.init_filters": 8, "model.spatial_dims": 3, "device": "cpu"}
    )
    from melba.models import build_model, count_parameters

    model = build_model(config).eval()
    assert count_parameters(model) > 0
    volume = torch.randn(1, 4, 24, 24, 32)  # divisible by 8: three SegResNet down-samples
    with torch.inference_mode():
        logits = model(volume)
    assert logits.shape == (1, 1, 24, 24, 32)

    window = build_sliding_window(config)
    assert window["roi"] == (128, 128, 128) and window["overlap"] == 0.5
    with torch.inference_mode():
        stitched = sliding_window_predict(
            model,
            volume,
            torch.device("cpu"),
            roi=(16, 16, 16),
            overlap=0.5,
            amp=False,
        )
    assert stitched.shape == logits.shape
    assert torch.isfinite(stitched).all()
    # windows are blended, not one pass copied over everything
    assert not torch.allclose(stitched, logits)


def test_sliding_window_config_validation() -> None:
    config = load_config(CONFIG)
    assert build_sliding_window(config) == {
        "roi": (128, 128, 128),
        "overlap": 0.5,
        "mode": "gaussian",
        "sw_batch_size": 1,
        "padding_mode": "constant",
    }
    disabled = load_config(CONFIG, {"inference.sliding_window": False})
    assert build_sliding_window(disabled) is None
    with pytest.raises(ValueError, match="overlap"):
        build_sliding_window(
            load_config(CONFIG, {"inference.sliding_window": {"overlap": 1.5}})
        )
    with pytest.raises(ValueError, match="Unknown key"):
        build_sliding_window(
            load_config(CONFIG, {"inference.sliding_window": {"overlap": 0.5, "bogus": 1}})
        )



def _cache_case(tmp_path: Path, brats_root: Path, case_index: int = 0):
    from melba.data.brats import BraTSItem, preprocess_case

    cache_dir = tmp_path / "cache"
    case = f"BraTS2021_{case_index:05d}"
    item = BraTSItem(
        case_id=case, group_id=case, split="test", directory=brats_root / case
    )
    entry = preprocess_case(item, cache_dir=cache_dir, margin=4)
    return item, cache_dir, entry


def test_cached_case_matches_the_nifti_at_the_same_crop(tmp_path: Path, brats_root: Path) -> None:
    """The float16 cache must reproduce the uncached arrays inside the crop."""
    from melba.data.brats import load_case_for_pipeline

    item, cache_dir, entry = _cache_case(tmp_path, brats_root)
    cached, cached_mask, cached_spacing, meta = load_case_for_pipeline(
        item, use_cache=True, cache_dir=cache_dir
    )
    full, full_mask, full_spacing, full_meta = load_case_for_pipeline(item, use_cache=False)

    assert meta["source"] == "cache" and full_meta["source"] == "nifti"
    assert cached_spacing == pytest.approx(full_spacing)
    assert tuple(meta["original_shape"]) == full.shape[1:]  # 240x240x155 in the real data
    offsets = tuple(int(value) for value in meta["offsets"])
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, cached.shape[1:])
    )
    region = full[(slice(None), *slices)]
    # same crop, same values: float16 storage is the only difference
    np.testing.assert_allclose(cached, region, rtol=2e-3, atol=5e-3)
    assert np.array_equal(cached_mask, full_mask[slices])
    # the ET mask is a subset of the brain (which is exactly the non-zero region)
    brain = cached[0] != 0
    assert np.all(cached_mask.astype(bool) <= brain)
    # placing the cached volume back reproduces the full volume (air is exactly 0)
    from melba.data.brats import place_on_full_grid

    restored = place_on_full_grid(cached, offsets=offsets, original_shape=full.shape[1:])
    np.testing.assert_allclose(restored, full, rtol=2e-3, atol=5e-3)
    assert np.all(restored[:, : offsets[0]] == 0.0)


def test_predictions_placed_back_give_identical_metrics(tmp_path: Path, brats_root: Path) -> None:
    """Cropped-grid metrics equal full-grid metrics once the prediction is placed back."""
    from melba.data.brats import load_case_for_pipeline, place_on_full_grid
    from melba.metrics import MetricSettings, compute_case_metrics

    item, cache_dir, _entry = _cache_case(tmp_path, brats_root)
    cached, mask, spacing, meta = load_case_for_pipeline(
        item, use_cache=True, cache_dir=cache_dir
    )
    full, full_mask, full_spacing, full_meta = load_case_for_pipeline(item, use_cache=False)
    offsets = tuple(int(value) for value in meta["offsets"])

    # a deliberately imperfect prediction: the reference mask, eroded by one voxel
    eroded = (mask[1:-1, 1:-1, 1:-1]).copy()
    eroded[eroded > 0] = 1.0
    prediction = np.zeros_like(mask)
    prediction[1:-1, 1:-1, 1:-1] = eroded

    settings = MetricSettings(names=("dice", "hd95", "assd", "boundary_f1"))
    cropped_metrics = compute_case_metrics(
        prediction, mask, spacing=spacing, settings=settings
    )
    prediction_full = place_on_full_grid(
        prediction[None], offsets=offsets, original_shape=full.shape[1:]
    )[0]
    target_full = place_on_full_grid(
        full_mask[None], offsets=offsets, original_shape=full.shape[1:]
    )[0]
    assert np.array_equal(target_full, full_mask)  # padding restores the reference exactly
    full_metrics = compute_case_metrics(
        prediction_full, target_full, spacing=full_spacing, settings=settings
    )
    for name in ("dice", "hd95", "assd", "boundary_f1"):
        left, right = cropped_metrics[name], full_metrics[name]
        if left != left:  # NaN == NaN for a degenerate pair
            assert right != right
        else:
            assert left == pytest.approx(right, rel=1e-6, abs=1e-9), name

