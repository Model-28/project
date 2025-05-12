# project MODEL 28

## Dataset Description
Our group will be using the Intel Image Classification Dataset, a publicly available Kaggle collection of approximately 24,000 RGB images at a uniform 150×150‐pixel resolution. The data are organized into six classes‐named subdirectories—buildings, forest, glacier, mountain, sea, and street—each assigned integer labels 0 through 5. The original download provides separate seg_train and seg_test folders (roughly 14,000 training and 3,000 testing images), and we will further split the training set 80/20 for validation. All files are in JPEG format, balanced across classes, and require no additional annotation, making this dataset straightforward to load, preprocess, and feed into our flattened‐pixel (67,500‐feature) and CNN pipelines.

## Data Preprocessing

Before training any model, you need to preprocess the raw image data. Follow the steps below to generate ready-to-use `.npy` files for your assignments.

---

### Download the data from Kaggle

Download the **Intel Image Classification Dataset** from Kaggle:  
-> [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

After downloading:
- Unzip the dataset
- You will get a folder named `archive/`
- **Move the entire `archive/` folder into the root directory of this project**

---

### 1: Go to the project root directory

Open your terminal and make sure you are in the **root directory of the project**.  
To verify, run:

```bash
ls
```

If you see the following folders and files:

```
archive/  data_preprocess/  figures/  models/  notebooks/  README.md
```

then you are in the correct place.

---

### 2. Run the preprocessing script

Run this command:

```bash
python data_preprocess/preprocess.py
```

This will:
- Traverse all class-named folders under `archive/seg_train/seg_train`
- Resize every image to 150×150
- Normalize pixel values to the range [0, 1]
- Flatten each image into a 1D array
- Save two output files in the current directory:
  - `X.npy` — the image data
  - `y.npy` — the corresponding labels

---

### What are X.npy and y.npy?

| File     | Shape                      | Contents                                  | Used for                             |
|----------|----------------------------|-------------------------------------------|--------------------------------------|
| `X.npy`  | `(N, 150×150×3)` → `(N, 67500)` | Flattened, normalized RGB image data     | Input features for MLP, LogReg, CNN  |
| `y.npy`  | `(N,)`                     | Integer class labels (e.g., 0–5)          | Target labels for classification     |

> These arrays will be the basis for your exploratory data analysis (EDA), training, validation, and testing.

---

### 3. Load the `.npy` files in your script

Once you've run the command above, you can load the data in any script like this:

```python
import numpy as np

# load preprocessed features and labels
X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)   # (N, 67500)
print("y shape:", y.shape)   # (N,)
```

- You can now use `X` and `y` directly in your:
  - **Logistic Regression models**
  - **Multi-Layer Perceptrons (MLP)**
  - **Convolutional Neural Networks (CNN)** (after reshaping)
  - **EDA visualizations**
  - **Train/test splits and k-fold cross-validation**

---

Once this step is complete, you're ready to move on to model building and analysis. 🚀
