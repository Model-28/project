# flask_cnn.py

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "best_model.keras")
RAW_DIR     = os.path.abspath(os.path.join(
                 BASE_DIR, "..", "..", "..", "archive", "seg_train", "seg_train"
              ))
IMG_SIZE    = (150, 150)
PORT        = 3000

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def preprocess_image(stream):
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # shape (1, H, W, 3)

# ─── APP SETUP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)

# Build class name list from your folder names
class_names = sorted(
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
)
print("Class names:", class_names)

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            X     = preprocess_image(file.stream)
            probs = model.predict(X)
            idx   = int(np.argmax(probs, axis=1)[0])
            pred  = class_names[idx]
            conf  = float(probs[0, idx])

            prediction = pred
            confidence = f"{conf:.3f}"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

# ─── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)