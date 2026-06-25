import os
import pandas as pd
from modules.module3_vision.dataset import build_split_index, RAW_ROOT

df = build_split_index(force_rebuild=True)

print("Checking real file existence on disk (this scans ~27k files, may take a minute)...")
df["exists"] = df["image_path"].apply(lambda p: os.path.exists(os.path.join(RAW_ROOT, p)))

print()
print("=== Per-dataset availability ===")
for ds in df["source_dataset"].unique():
    subset = df[df["source_dataset"] == ds]
    print(f"{ds}: {subset['exists'].sum()} / {len(subset)} images available")

print()
real_df = df[df["exists"]]
print(f"Total usable images right now: {len(real_df)} / {len(df)}")
print()

train_real = real_df[real_df["final_split"] == "train"]
test_real = real_df[real_df["final_split"] == "test"]
print(f"Usable train samples: {len(train_real)}")
print(f"Usable test samples: {len(test_real)}")
print()

counts = train_real["label_collapsed"].value_counts()
strong = counts[counts.index != "UNCLASSIFIED"]
print(f"Classes with at least 1 usable training image: {len(strong)} / 127")
print()
print("Weakest 10 (by REAL available images):")
print(strong.sort_values().head(10))
print()
print("Classes with ZERO usable images (effectively untrainable right now):")
zero_classes = set(df[df["final_split"]=="train"]["label_collapsed"].unique()) - set(strong.index) - {"UNCLASSIFIED"}
print(zero_classes)