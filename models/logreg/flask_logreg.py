# app.py

import io
import numpy as np
from PIL import Image
from joblib import load
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template

# ——— CONFIG ———
MODEL_PATH   = "train_logreg.h5"   # <-- your new Keras model file
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

    # load Keras model and your label encoder
    model = load_model(MODEL_PATH)
    le    = load(ENCODER_PATH)

    @app.route("/", methods=["GET", "POST"])
    def home():
        prediction = None
        confidence = None

        if request.method == "POST":
            file = request.files.get("image")
            if file:
                X = preprocess_image(file.stream)

                # get prediction probabilities
                probs = model.predict(X)
                idx   = int(np.argmax(probs, axis=1)[0])
                prob  = float(np.max(probs))

                prediction = le.inverse_transform([idx])[0]
                confidence = f"{prob:.3f}"

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence
        )

    app.run(host="0.0.0.0", port=PORT, debug=True)

if __name__ == "__main__":
    main()