# complete implementation of intel classification dataset models
# we will have all three models here. 

# app.py

import os, io
import numpy as np
from PIL import Image
import tensorflow as tf
from joblib import load
from flask import Flask, request, render_template

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH      = os.path.join(BASE_DIR, "best_model.keras")
MLP_MODEL_PATH      = os.path.join(BASE_DIR, "best_mlp.h5")
LOGREG_MODEL_PATH   = os.path.join(BASE_DIR, "train_logreg.h5")
RAW_DIR             = os.path.abspath(os.path.join(
                        BASE_DIR, "..", "..", "archive", "seg_train", "seg_train"
                     ))
IMG_SIZE_CNN        = (150, 150)
IMG_SIZE_MLP        = (150, 150)
IMG_SIZE_LOGREG     = (64, 64)
PORT                = 3000

# ─── LOAD MODELS & ENCODER ───────────────────────────────────────────────────────
cnn_model   = tf.keras.models.load_model(CNN_MODEL_PATH)
mlp_model   = tf.keras.models.load_model(MLP_MODEL_PATH)
logreg_model= tf.keras.models.load_model(LOGREG_MODEL_PATH)

# shared class names for CNN & MLP (folder names)
class_names = sorted(
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
)

# ─── PREPROCESS FUNCTIONS ───────────────────────────────────────────────────────
def preprocess_cnn(stream):
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE_CNN)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def preprocess_mlp(stream):
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE_MLP)
    arr = np.array(img, dtype=np.float32)
    return arr.flatten()[np.newaxis, ...]

def preprocess_logreg(stream):
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE_LOGREG)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()[np.newaxis, ...]

# ─── FLASK SETUP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    cnn_pred = mlp_pred = logreg_pred = None
    cnn_conf = mlp_conf = logreg_conf = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            data = file.read()

            # CNN inference
            Xc = preprocess_cnn(io.BytesIO(data))
            pc = cnn_model.predict(Xc)
            ic = int(np.argmax(pc, axis=1)[0])
            cnn_pred = class_names[ic]
            cnn_conf = f"{float(pc[0,ic]):.3f}"

            # MLP inference
            Xm = preprocess_mlp(io.BytesIO(data))
            pm = mlp_model.predict(Xm)
            im = int(np.argmax(pm, axis=1)[0])
            mlp_pred = class_names[im]
            mlp_conf = f"{float(pm[0,im]):.3f}"

            # Logistic regression inference
            Xl = preprocess_logreg(io.BytesIO(data))
            pl = logreg_model.predict(Xl)
            il = int(np.argmax(pl, axis=1)[0])
            logreg_pred = class_names[il]
            logreg_conf = f"{float(np.max(pl)):.3f}"

    return render_template(
        "index.html",
        cnn_pred=cnn_pred,   cnn_conf=cnn_conf,
        mlp_pred=mlp_pred,   mlp_conf=mlp_conf,
        logreg_pred=logreg_pred, logreg_conf=logreg_conf
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)