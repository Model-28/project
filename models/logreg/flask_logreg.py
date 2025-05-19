# app.py

import io
import numpy as np
from PIL import Image
from joblib import load
from flask import Flask, request, render_template

# ——— CONFIG ———
MODEL_PATH   = "intel_clf.joblib"
ENCODER_PATH = "intel_le.joblib"
IMG_SIZE     = (64, 64)
PORT         = 3000

# ——— PREPROCESSING ———
def preprocess_image(stream):
    img = Image.open(stream).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).flatten().astype(np.float32) / 255.0
    return arr.reshape(1, -1)

def main():
    app = Flask(__name__)
    clf = load(MODEL_PATH)
    le  = load(ENCODER_PATH)

    @app.route("/", methods=["GET", "POST"])
    def home():
        prediction = None
        confidence = None

        if request.method == "POST":
            file = request.files.get("image")
            if file:
                X = preprocess_image(file.stream)
                idx = int(clf.predict(X)[0])
                prob = float(np.max(clf.predict_proba(X)))
                prediction = le.inverse_transform([idx])[0]
                confidence = f"{prob:.3f}"

        return render_template("index.html",
                               prediction=prediction,
                               confidence=confidence)

    app.run(host="0.0.0.0", port=PORT, debug=True)

if __name__ == "__main__":
    main()