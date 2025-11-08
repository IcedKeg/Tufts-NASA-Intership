# test_lightglue_import.py
# Purpose: Verify LightGlue + dependency imports

try:
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from lightglue import LightGlue, SuperPoint, viz2d

    print("✅ All imports successful. LightGlue and dependencies are ready.")
except ImportError as e:
    print("❌ Import failed:", e)
