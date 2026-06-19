import torch

ckpt_path = "models/checkpoints/Swin_MC_best_model.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    print("Top-level keys in checkpoint dict:", list(ckpt.keys()))
elif isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
    print("Top-level keys in checkpoint dict:", list(ckpt.keys()))
else:
    state_dict = ckpt
    print("Checkpoint is a raw state_dict")

print(f"\nTotal parameter tensors: {len(state_dict)}")

print("\n--- First 10 keys (patch embed / stem) ---")
for k in list(state_dict.keys())[:10]:
    print(k, state_dict[k].shape)

print("\n--- Last 10 keys (classification head) ---")
for k in list(state_dict.keys())[-10:]:
    print(k, state_dict[k].shape)

for k, v in state_dict.items():
    if "patch_embed.proj.weight" in k:
        print(f"\nPatch embed conv weight shape: {v.shape}")

for k, v in state_dict.items():
    if "absolute_pos_embed" in k or "pos_embed" in k:
        print(f"Position embedding shape: {v.shape}")

for k, v in state_dict.items():
    if "head" in k.lower() or "fc" in k.lower() or "classifier" in k.lower():
        print(f"Possible head layer: {k}  shape={v.shape}")