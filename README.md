# Fashion-MNIST Classification — ANN vs CNN

**A side-by-side comparison of a plain neural network and a convolutional network on the 10-class Fashion-MNIST image dataset.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This is an image-classification notebook that takes 28×28 grayscale photos of clothing — shirts, trousers, sneakers, bags, ankle boots and so on — and sorts each one into the correct category out of ten. It uses Fashion-MNIST, the drop-in replacement for the classic digit MNIST that's harder to crack because the classes (e.g. pullover vs coat vs shirt) actually look alike.

The point of the notebook isn't just to get a number. It builds two models on the same data and compares them: a fully-connected artificial neural network (ANN) as a baseline, then a convolutional neural network (CNN) to show what convolution buys you on image data. The CNN lands at **91.2% test accuracy** versus the ANN's **~85%**, which is the lesson the notebook is built to demonstrate.

It was written as a first deep-learning exercise (the introductory image-classification task in a guided ML curriculum), so it stays deliberately readable end to end — load, look, normalize, train, score, repeat.

## Key Features

- Loads Fashion-MNIST directly through the Keras built-in dataset loader — 60,000 training images and 10,000 test images, all 28×28 grayscale.
- Quick visual EDA: a small `plot()` helper renders individual samples with `matshow` so you can eyeball what each class looks like before training.
- Class-balance check via a pandas `value_counts()` — confirms all ten classes are evenly represented (exactly 6,000 images each in the training set).
- Pixel normalization from the 0–255 range down to 0–1 before training.
- Two complete model builds with the Keras Sequential API:
  - An **ANN** baseline (Flatten + two large Dense layers).
  - A **CNN** (two Conv/MaxPool blocks + a dense head).
- Full evaluation: per-class precision / recall / F1 via `classification_report`, plus a Seaborn confusion-matrix heatmap.
- Spot-checks individual CNN predictions against ground-truth labels for the first ten test images.

## How It Works

The whole pipeline lives in a single notebook, `Fashion Mnist.ipynb`, and runs top to bottom.

### Data loading

Data comes from `tensorflow.keras.datasets.fashion_mnist.load_data()` — not from any file in the repo. That call returns the standard split: `X_train` of shape `(60000, 28, 28)`, `X_test` of `(10000, 28, 28)`, with matching integer label arrays. Labels are 0–9, one per clothing category.

### EDA and preprocessing

Before any model, the notebook plots a handful of training images with a small helper that shows the pixel grid and prints the label, so you can sanity-check that label 5 really is a sandal, label 9 an ankle boot, etc. A pandas `value_counts()` on the labels confirms the dataset is perfectly balanced — 6,000 examples per class. Preprocessing is one line: divide both `X_train` and `X_test` by 255 to scale pixels into [0, 1], which keeps the gradients well-behaved during training.

### Model 1 — ANN baseline

```
Flatten(28×28 → 784)
Dense(3000, relu)
Dense(1000, relu)
Dense(10, sigmoid)
```

Compiled with the SGD optimizer and `sparse_categorical_crossentropy` loss, trained for 5 epochs. It's intentionally a no-convolution baseline: flatten the image into a 784-length vector and let two wide dense layers do the work. By the last epoch it reaches **87.2% training accuracy**, and on the held-out test set it scores about **85% accuracy** — the `classification_report` gives the full per-class breakdown (strong on trousers, bags and ankle boots; weakest on the shirt class, which is the usual Fashion-MNIST troublemaker).

### Model 2 — CNN

```
Conv2D(32, 3×3, relu)  →  MaxPool(2×2)
Conv2D(64, 3×3, relu)  →  MaxPool(2×2)
Flatten
Dense(128, relu)
Dense(10, softmax)
```

Compiled with the Adam optimizer and the same loss, trained for 10 epochs with a batch size of 32. The convolutional layers learn local edge/texture filters instead of treating every pixel independently, which is what closes the gap on image data. Training accuracy climbs to **96.0%** by epoch 10, and `cnn.evaluate()` reports **91.2% test accuracy** (test loss 0.2903) — a clear jump over the ANN baseline on the same split.

### Evaluation

Both models are scored with scikit-learn's `classification_report` for precision/recall/F1 per class, and a 10×10 confusion matrix is drawn as a Seaborn heatmap (`fmt='d'`, `cmap='Blues'`). The notebook also prints the CNN's raw softmax probability vectors for the first ten test images and compares the `argmax` predictions against the true labels — a small manual check that the model is doing what the aggregate score says it is.

## Results / Highlights

| Model | Optimizer | Epochs | Train accuracy | Test accuracy |
|-------|-----------|--------|----------------|----------------|
| ANN (baseline) | SGD | 5 | 87.2% | ~85% |
| CNN | Adam | 10 | 96.0% | **91.2%** |

- The CNN improves test accuracy by roughly 6 points over the dense baseline on the same data — the core takeaway of the project.
- ANN per-class F1 ranges from 0.58 (shirts, the hardest class) up to 0.97 (trousers); test loss for the CNN is 0.2903.
- Dataset is fully balanced: 6,000 training images per class across all ten categories.

## Tech Stack

- **Language:** Python (Jupyter Notebook)
- **Deep learning:** TensorFlow / Keras (Sequential API, `Conv2D`, `MaxPooling2D`, `Dense`, `Flatten`)
- **Data / ML:** NumPy, pandas, scikit-learn (`classification_report`, `confusion_matrix`)
- **Visualization:** Matplotlib, Seaborn

## Getting Started

### Prerequisites
- Python 3.x
- pip
- Jupyter (notebook or lab)

### Installation
```bash
git clone https://github.com/DCode-v05/Fashion-Mnist.git
cd Fashion-Mnist
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Running
```bash
jupyter notebook "Fashion Mnist.ipynb"
```

No dataset download is required — the notebook pulls Fashion-MNIST through the Keras loader on first run.

## Usage

Open `Fashion Mnist.ipynb` and run the cells top to bottom. The flow is: load data → plot a few samples → check class balance → normalize → train the ANN and read its classification report → train the CNN and evaluate it → inspect sample predictions → view the confusion-matrix heatmap.

To experiment, the obvious knobs are the layer sizes, the optimizer choice, the epoch count and the CNN batch size — all set in the model-build cells. Swapping the ANN's output activation, adding dropout, or training the CNN longer are natural next steps if you want to push the numbers further.

## Project Structure

```
Fashion-Mnist/
├── Fashion Mnist.ipynb        # The full workflow: load, EDA, normalize, ANN, CNN, evaluation
├── README.md                  # Project description
├── train-images-idx3-ubyte    # Fashion-MNIST training images, raw IDX binary (bundled artifact)
├── train-labels-idx1-ubyte    # Training labels, IDX binary
├── t10k-images-idx3-ubyte     # Test images, IDX binary
├── t10k-labels-idx1-ubyte     # Test labels, IDX binary
├── fashion-mnist_test.csv     # Test set in CSV form (~22 MB)
└── fashion-mnist_train.csv    # Placeholder stub (not a full dataset file)
```

Note: the notebook itself loads data via the Keras `fashion_mnist.load_data()` loader, so the IDX and CSV files ship with the repo as copies of the dataset but aren't read by the code. `fashion-mnist_train.csv` is a tiny stub rather than the actual training data.

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
