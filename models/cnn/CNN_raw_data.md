# CNN Data Analysis Plan (Using Raw Images)

## Overview 
1. **Load** images directly from folders (no `.npy` files)
2. **Use TensorFlow's ImageDataGenerator** to handle preprocessing
3. **Split** into training/validation sets automatically
4. **Build** the same CNN model architecture
5. **Train** using image batches (better for memory)
6. **Evaluate** on test images from separate folder
7. **Visualize** predictions and performance


---

## 1. Loading Raw Images (no reshaping needed)

We can use TensorFlow's `ImageDataGenerator` which handles:
- Reading images directly from folders
- Automatic resizing/normalization
- Creating labels from folder names

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to our folders
train_dir = 'archive/seg_train/seg_train'
test_dir = 'archive/seg_test/seg_test'

# Create generators with normalization
train_datagen = ImageDataGenerator(rescale=1./255, 
                                  validation_split=0.2) # 80/20 split
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow images directly from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    subset='training') # Uses 80% of data

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    subset='validation') # Uses 20% of data

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    shuffle=False) # Don't shuffle test data

```


2. Building the Same CNN Model
The model architecture stays identical, but now it receives images directly:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # First convolution layer
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    
    # Second convolution layer  
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    # Third convolution layer
    layers.Conv2D(128, (3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])
```

3. Compile and Train with Generators
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train using image batches
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32, # Number of batches
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32)

```


4. Evaluating on Test Data
```python
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

test_predictions = model.predict(test_generator)
test_pred_classes = np.argmax(test_predictions, axis=1)

cm = confusion_matrix(test_generator.classes, test_pred_classes)
```

5. Visualizing Results
Training History:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
```

Sample Predictions:

```python
# Get a batch of test images
test_images, test_labels = next(test_generator)

# Predict and display
predictions = model.predict(test_images)
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[i])
    plt.title(f"Pred: {np.argmax(predictions[i])}, True: {test_labels[i]}")
    plt.axis('off')
plt.show()
```