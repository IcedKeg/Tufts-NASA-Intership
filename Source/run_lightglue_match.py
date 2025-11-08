# run_lightglue_match.py
# Purpose: Perform LightGlue feature matching between Ground and Satellite images.
import numpy as np
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

    # --- Extract valid keypoints ---
    kpts0 = matches["keypoints0"]
    kpts1 = matches["keypoints1"]
    matches0 = matches["matches0"]

    # --- Broadcast-safe filter ---
    valid = matches0 > -1
    mkpts0_valid = kpts0[valid]
    mkpts1_valid = kpts1[matches0[valid]]

    # --- Move to CPU + NumPy ints ---
    mkpts0_np = mkpts0_valid.detach().cpu().numpy().astype(int)
    mkpts1_np = mkpts1_valid.detach().cpu().numpy().astype(int)

    # --- Build side-by-side canvas ---
    vis0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    vis1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    h = max(vis0.shape[0], vis1.shape[0])
    w0, w1 = vis0.shape[1], vis1.shape[1]
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:vis0.shape[0], :w0] = vis0
    canvas[:vis1.shape[0], w0:w0 + w1] = vis1
    offset = np.array([w0, 0], dtype=int)

    # --- Draw lines for valid matches ---
    for p0, p1 in zip(mkpts0_np, mkpts1_np):
        pt1 = (int(p0[0]), int(p0[1]))
        pt2 = tuple((p1 + offset).astype(int))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

    # --- Save visualization ---
    output_dir = r"C:\Users\ddkab\Documents\GitHub\Tufts-NASA-Intership\Results"
    os.makedirs(output_dir, exist_ok=True)
    g_name = os.path.splitext(os.path.basename(g))[0]
    s_name = os.path.splitext(os.path.basename(s))[0]
    output_path = os.path.join(output_dir, f"match_{g_name}_{s_name}.jpg")
    cv2.imwrite(output_path, canvas)
    print(f"💾 Saved match visualization → {output_path}")


