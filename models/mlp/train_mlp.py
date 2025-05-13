# here we train and implement our multi-layered perceptron model

import sys, os

# make sure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocess import preprocess

import os
import numpy as np
from PIL import Image

def main():
    raw_dir = os.path.join(PROJECT_ROOT, "archive", "seg_train", "seg_train")
    # this print proves main() is running
    print("about to call preprocess()")
    preprocess(
       input_dir=raw_dir,
       target_size=(150,150),
       out_X="X.npy",
       out_y="y.npy"
    )
    print("back in train_mlp after preprocess()")

if __name__=="__main__":
	main()

