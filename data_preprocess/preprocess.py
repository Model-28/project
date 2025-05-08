# preprocess.py

import os
from collections import Counter
from PIL import Image
import numpy as np

def preprocess(input_dir: str,
               target_size: tuple[int,int] = (150,150),
               out_X: str = "X.npy",
               out_y: str = "y.npy"):
    """
    1) Finds all class‑subfolders in `input_dir`
    2) Resizes every image to `target_size`
    3) Flattens and normalizes into two .npy files: features X and labels y
    """
    print("🚀 preprocess() has started!")
    # --- find class names (only directories) ---
    classes = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    classes.sort()
    print("Classes:", classes)

    # --- resize all images in place ---
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            path = os.path.join(root, fname)
            img = Image.open(path).convert("RGB")
            if img.size != target_size:
                img = img.resize(target_size, resample=Image.LANCZOS)
                img.save(path)
    print(f"Resized all images to {target_size}")

    # --- flatten into arrays ---
    X_list, y_list = [], []
    for label, cls in enumerate(classes):
        cls_path = os.path.join(input_dir, cls)
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img = Image.open(os.path.join(cls_path, fname)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X_list.append(arr.flatten())
            y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)

    # --- save to disk ---
    np.save(out_X, X)
    np.save(out_y, y)
    print(f"Saved X.shape={X.shape} → {out_X}")
    print(f"Saved y.shape={y.shape} → {out_y}")
    print("✅ preprocess() has finished!")

if __name__=="__main__":
    import argparse

    # set your one‑time defaults here:
    DEFAULT_INPUT = "archive/seg_train/seg_train"
    DEFAULT_SIZE  = (150,150)
    DEFAULT_X     = "X.npy"
    DEFAULT_Y     = "y.npy"

    p = argparse.ArgumentParser(
       description="Resize, flatten and save image data to .npy")
    p.add_argument("-i","--input_dir",
                   default=DEFAULT_INPUT,
                   help=f"path to folder of class‑subfolders (default: %(default)s)")
    p.add_argument("-s","--size", nargs=2, type=int,
                   default=DEFAULT_SIZE,
                   help=f"target width height (default: %(default)s)")
    p.add_argument("-x","--out_X",
                   default=DEFAULT_X,
                   help=f"where to save features .npy (default: %(default)s)")
    p.add_argument("-y","--out_y",
                   default=DEFAULT_Y,
                   help=f"where to save labels .npy (default: %(default)s)")
    args = p.parse_args()

    preprocess(
      input_dir=args.input_dir,
      target_size=tuple(args.size),
      out_X=args.out_X,
      out_y=args.out_y
    )