import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 1) Load your trained MLP
model = load_model("best_mlp.h5")

# 2) Declare your classes in the exact order used during training:
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 3) Load a sample image to test
img = Image.open("../../archive/seg_test/seg_test/buildings/20057.jpg")\
           .convert("RGB")\
           .resize((150,150))

# 4) Preprocess exactly as you did in Flask  
x = np.array(img, dtype=np.float32)  
x = x.flatten()[None, :]           # shape (1, 67500)

# 5) Get raw probabilities and print them  
probs = model.predict(x)           # shape (1,6)  
print("Raw probs:", probs)  

# 6) Decode the top prediction  
idx = int(np.argmax(probs, axis=1)[0])  
print("Predicted class:", class_names[idx])  
print("Confidence:", probs[0, idx])