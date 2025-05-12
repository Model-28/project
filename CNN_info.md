# CNN Data Analysis Plan

## Overview 
1. **Load** our preprocessed image data (`X.npy`, `y.npy`)
2. **Reshape** our flattened data back into 3D image tensors
3. **Split** the dataset into training, validation, and testing sets
4. **Build** a CNN model using layers like Conv2D, MaxPooling, and Dense
5. **Train** the CNN on training data and track accuracy/loss
6. **Evaluate** the model on validation/test sets
7. **Visualize** predictions and performance

---

## 1. Load Preprocessed Data

We already ran the script that resized and normalized images, and saved them as `.npy` files.

```python
import numpy as np

X = np.load("X.npy")  # shape: (N, 67500)
y = np.load("y.npy")  # shape: (N,)
```

At this point, each image is still **flattened** (1D shape), but CNNs need the original **3D structure** (150 x 150 x 3), so we will reshape the images.

---

## 2. Reshape to 3D Images

We'll reshape our `X` array back to 3D format so the CNN can treat each image like an actual image.

```python
X = X.reshape(-1, 150, 150, 3)  # -1 means "infer batch size"
```

Now each image is in `[height, width, channels]` format.

---

## 3. Train/Validation/Test Split

We already have test images in `archive/seg_test/seg_test`. From the training data, we'll make an 80/20 train/val split:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

We split into **training** and **validation** sets (not test) so we can tune the model safely without bias. The **training set** is for learning, the **validation set** helps us evaluate during training and make improvements, and the **test set** (which we use only once at the end) gives a final, unbiased accuracy. This prevents overfitting and keeps our evaluation fair.

---

## 4. Build the CNN Model

Here’s a simple CNN using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 output classes
])
```

Layers in the CNN:
- `Conv2D`: Detects features (like edges or corners)
- `MaxPooling2D`: Downsamples image to reduce size and overfitting
- `Flatten`: Converts 3D data to 1D
- `Dense`: Fully connected layers
- `softmax`: Gives probability for each class (like “this image is 90% likely a mountain”)

---

## 5. Compile the Model

Now we tell the model how to learn (optimizer/loss function) and what to track (accuracy).

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

- **Adam** (Adaptive Moment Estimation) is a popular optimization algorithm used to update the model's weights during training. "Optimizer" is responsible for adjusting the model's parameters (weights) to minimize the error (or loss) after each prediction.
- **Sparse categorical crossentropy** is a loss function used in multi-class classification tasks where labels are integers, helping the model measure how far its predictions are from the correct class.
- **Accuracy** is a common metric that measures the percentage of correct predictions the model makes. 

---

## 6. Train the Model

Now we let the model learn from the training set and check how it performs on the validation set.

```python
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val))
```

---

## 7. Evaluate and Visualize

After training, we'll look at how the model performs on the validation set.

```python
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")
```

We can also visualize predictions:

```python
import matplotlib.pyplot as plt

# sample prediction
sample_image = X_val[0]
plt.imshow(sample_image)
plt.title(f"Predicted: {np.argmax(model.predict(sample_image[np.newaxis]))}, Actual: {y_val[0]}")
```

---


## Plot Training Curves

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy over Epochs")
plt.show()
```

---
