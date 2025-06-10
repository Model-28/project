# flask_mlp.py

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# ——— CONFIG ———
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "best_mlp.h5") # this is the trained model right now a little more accurate than logreg
CLASS_NAMES_PATH  = os.path.join(BASE_DIR, "y.npy")  # saved list of string class names

## --------------- EDIT THIS TO YOUR DATA PATH ---------------------------------------
RAW_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "archive", "seg_train", "seg_train")) 
## -----------------------------------------------------------------------------------

IMG_SIZE          = (150, 150) 
PORT              = 3000

# ——— HELPERS ———
def preprocess_image(stream):
    """
    Open an image stream, convert to RGB, resize to IMG_SIZE,
    normalize pixels to [0,1], and flatten for prediction.
    """
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

# ——— FLASK APP SETUP ———
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names (Python list of strings)
# Re-create the list of string class names by listing & sorting your folders:
class_names = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
])
print("Class names:", class_names)

@app.route("/", methods=["GET"])
def index():
    # Render the upload & predict page
    return render_template("index.html")

@app.route("/predict_file", methods=["POST"])
def predict_file():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No file uploaded"), 400

    # Preprocess and predict
    X = preprocess_image(file.stream)
    probs = model.predict(X)
    idx = int(np.argmax(probs, axis=1)[0])

    return jsonify(
        prediction=class_names[idx],
        confidence=f"{float(probs[0, idx]) * 100:.2f}%"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
