"""
Module 3 — dataset.py
Builds the train/test split (collapsing rare classes into UNCLASSIFIED,
filling the missing test split for Fitzpatrick17k/PAD-UFES-20/SCIN while
leaving DermaCon-IN's existing subject-level split untouched), caches the
result, and exposes a PyTorch Dataset over it.

Verified live (June 2026) before writing this file:
- Raw parquet: data/processed/skintel_bedrock.parquet (27,139 records)
- image_path is relative to data/raw/ (e.g. "fitzpatrick17k/images/xxx.jpg")
- Only dermaconin has a real split; the other 3 datasets are 100% 'train'
  in the raw file — this script fixes that gap, it doesn't touch dermaconin
- 'label' (255 unique) is the real diagnosis target, not 'main_class' (18,
  a coarse mechanism category) or 'sub_class' (19, mostly null)
- <15-sample labels (128 of 255) collapse into one UNCLASSIFIED class,
  computed on GLOBAL counts so dermaconin's side of the data stays consistent
- 2 classes (Chickenpox, Alopecia Areata) have <2 samples outside dermaconin
  and are routed straight to train rather than crashing the stratified split
- Effective class count after collapse: 128
"""

import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

RAW_PARQUET = "data/processed/skintel_bedrock.parquet"
RAW_ROOT = "data/raw"
CACHE_PARQUET = "data/processed/module3_split_index.parquet"
LABEL_MAP_JSON = "data/processed/module3_label_to_idx.json"

MIN_SAMPLES_TO_KEEP = 15  # below this, collapse into UNCLASSIFIED
TEST_SIZE_FOR_UNSPLIT = 0.15
RANDOM_STATE = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _collapse_labels(df: pd.DataFrame) -> pd.Series:
    """Collapse rare labels into UNCLASSIFIED using GLOBAL counts across
    all 4 datasets — keeps the boundary consistent with dermaconin's side
    of the data, not just whatever subset happens to be passed in."""
    counts = df["label"].value_counts()
    keep = set(counts[counts >= MIN_SAMPLES_TO_KEEP].index)
    return df["label"].where(df["label"].isin(keep), "UNCLASSIFIED")


def build_split_index(force_rebuild: bool = False) -> pd.DataFrame:
    """Builds (or loads cached) the final train/test index with collapsed
    labels. Safe to call every run — cheap to load, expensive to rebuild."""
    if os.path.exists(CACHE_PARQUET) and not force_rebuild:
        return pd.read_parquet(CACHE_PARQUET)

    df = pd.read_parquet(RAW_PARQUET)

    # Filter to real files BEFORE collapsing/splitting — otherwise the
    # class-balanced sampler's floor/cap logic operates on nominal Parquet
    # counts, not real availability, and __getitem__'s missing-file retry
    # silently substitutes a DIFFERENT class's image into the gap, quietly
    # distorting the intended per-class balance.
    file_exists = df["image_path"].apply(lambda p: os.path.exists(os.path.join(RAW_ROOT, p)))
    n_before = len(df)
    df = df[file_exists].reset_index(drop=True)
    print(f"Filtered to real files on disk: {len(df)} / {n_before}")

    df["label_collapsed"] = _collapse_labels(df)

    dermacon = df[df["source_dataset"] == "dermaconin"].copy()
    dermacon["final_split"] = dermacon["split"]  # untouched, as documented

    others = df[df["source_dataset"] != "dermaconin"].copy()
    collapsed_counts = others["label_collapsed"].value_counts()

    # Classes with <2 samples in this subset can't be stratified —
    # route them straight to train rather than crash the split.
    unsplittable = set(collapsed_counts[collapsed_counts < 2].index)
    forced_train = others[others["label_collapsed"].isin(unsplittable)].copy()
    forced_train["final_split"] = "train"

    splittable = others[~others["label_collapsed"].isin(unsplittable)].copy()
    train_part, test_part = train_test_split(
        splittable,
        test_size=TEST_SIZE_FOR_UNSPLIT,
        stratify=splittable["label_collapsed"],
        random_state=RANDOM_STATE,
    )
    train_part["final_split"] = "train"
    test_part["final_split"] = "test"

    combined = pd.concat([dermacon, train_part, test_part, forced_train], ignore_index=True)

    os.makedirs(os.path.dirname(CACHE_PARQUET), exist_ok=True)
    combined.to_parquet(CACHE_PARQUET, index=False)

    # Save label_to_idx — sorted for determinism, so re-runs never reshuffle
    # class indices even if the cache gets rebuilt.
    classes = sorted(combined["label_collapsed"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    return combined


def load_label_to_idx() -> dict:
    if not os.path.exists(LABEL_MAP_JSON):
        build_split_index()  # ensures it gets created
    with open(LABEL_MAP_JSON) as f:
        return json.load(f)


def get_class_counts(df: pd.DataFrame, split: str = "train") -> pd.Series:
    """Per-class sample counts for the given split — used by the training
    loop for weighted focal loss / oversampling caps, not computed here."""
    subset = df[df["final_split"] == split]
    return subset["label_collapsed"].value_counts()


def _build_transform(split: str) -> T.Compose:
    if split == "train":
        # Augmentation only — no batch preprocessing (Module 2's documented
        # decision: feed raw images, augment, don't crop/inpaint/normalize
        # colour at the dataset level the way runtime inference does).
        return T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class SkinTelDataset(Dataset):
    def __init__(self, split: str, raw_root: str = RAW_ROOT, transform=None):
        assert split in ("train", "test")
        self.raw_root = raw_root
        self.split = split

        full_index = build_split_index()
        self.df = full_index[full_index["final_split"] == split].reset_index(drop=True)
        self.label_to_idx = load_label_to_idx()
        self.transform = transform or _build_transform(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Bounded retry, not recursion — some classes currently have a large
        # fraction of missing files (Fitzpatrick17k re-download in progress,
        # SCIN partially downloaded), so this needs to handle many
        # consecutive misses safely, not just one rare bad file.
        for _ in range(len(self.df)):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.raw_root, row["image_path"])
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                label_idx = self.label_to_idx[row["label_collapsed"]]
                return image, label_idx
            except (FileNotFoundError, OSError):
                idx = (idx + 1) % len(self.df)
        raise RuntimeError("No loadable image found anywhere in the dataset.")


if __name__ == "__main__":
    # Live smoke test — same principle as model.py: verify before trusting.
    train_ds = SkinTelDataset(split="train")
    test_ds = SkinTelDataset(split="test")
    print("train size:", len(train_ds))
    print("test size:", len(test_ds))
    print("num classes:", len(train_ds.label_to_idx))

    img, label = train_ds[0]
    print("sample image tensor shape:", img.shape)
    print("sample label idx:", label)