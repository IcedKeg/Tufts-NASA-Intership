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
