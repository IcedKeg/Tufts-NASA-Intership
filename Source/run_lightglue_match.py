# run_lightglue_match.py
# Purpose: Perform LightGlue feature matching between Ground and Satellite images.

import os
import cv2
import torch
from lightglue import LightGlue, SuperPoint

# -----------------------------
# ✅ Configure your dataset paths
# -----------------------------
GROUND_DIR = r"C:\Users\ddkab\Documents\GitHub\Tufts-NASA-Intership\DataSet\Ground"
SAT_DIR = r"C:\Users\ddkab\Documents\GitHub\Tufts-NASA-Intership\DataSet\Satalite"

# Collect image pairs
ground_imgs = sorted([os.path.join(GROUND_DIR, f) for f in os.listdir(GROUND_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
sat_imgs = sorted([os.path.join(SAT_DIR, f) for f in os.listdir(SAT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

if not ground_imgs or not sat_imgs:
    print("❌ No images found. Check your Ground and Satalite folders.")
    exit()

print(f"📂 Found {len(ground_imgs)} ground images and {len(sat_imgs)} satellite images.")
print("⚙️ Initializing SuperPoint + LightGlue models...")

# -----------------------------
# ✅ Load models (GPU)
# -----------------------------
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
matcher = LightGlue(features='superpoint').eval().cuda()

print("✅ Models loaded successfully!\n")

# -----------------------------
# 🧮 Matching Loop
# -----------------------------
for g, s in zip(ground_imgs, sat_imgs):
    print(f"🖼️ Matching: {os.path.basename(g)} ↔ {os.path.basename(s)}")

    img0, img1 = cv2.imread(g, cv2.IMREAD_GRAYSCALE), cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        print(f"⚠️ Skipping pair: could not load {g} or {s}")
        continue

    # Convert to Torch tensors
    tensor0 = torch.from_numpy(img0)[None, None].float().cuda() / 255.
    tensor1 = torch.from_numpy(img1)[None, None].float().cuda() / 255.

    # Extract features
    feats0 = extractor({"image": tensor0})
    feats1 = extractor({"image": tensor1})

    # Match features
    matches = matcher({"image0": feats0, "image1": feats1})
    print("Returned keys:", matches.keys())
    # 🔍 Inspect what LightGlue actually returned due to keypoint error with "keypoints0"
for k, v in matches.items():
    print(k, type(v))

    mkpts0, mkpts1 = matches["matches0"], matches["matches1"]

    print(f"✅ Matches found: {len(mkpts0)}")

print("\n🏁 All image pairs processed successfully!")

# ---- Visualization & Save ----
import numpy as np, cv2, os

# Convert grayscale to RGB
vis0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
vis1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

# Create a combined canvas
h = max(vis0.shape[0], vis1.shape[0])
canvas = np.zeros((h, vis0.shape[1] + vis1.shape[1], 3), dtype=np.uint8)
canvas[:vis0.shape[0], :vis0.shape[1]] = vis0
canvas[:vis1.shape[0], vis0.shape[1]:vis0.shape[1] + vis1.shape[1]] = vis1

# Offset for second image (so matches align correctly)
offset = np.array([vis0.shape[1], 0])

# Draw lines for each match
for p0, p1 in zip(mkpts0.astype(int), mkpts1.astype(int)):
    pt1 = tuple(p0[::-1])
    pt2 = tuple((p1 + offset)[::-1])
    cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

# -----------------------------
# ✅ SAVE the visualized result
# -----------------------------
output_dir = r"C:\Users\ddkab\Documents\GitHub\Tufts-NASA-Intership\Results"
os.makedirs(output_dir, exist_ok=True)

output_name = f"match_{os.path.splitext(os.path.basename(g))[0]}_{os.path.splitext(os.path.basename(s))[0]}.jpg"
output_path = os.path.join(output_dir, output_name)

cv2.imwrite(output_path, canvas)
print(f"💾 Saved match visualization → {output_path}")
