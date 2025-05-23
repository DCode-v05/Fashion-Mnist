# 👗 Fashion-MNIST Classification

[🔗 GitHub Repository](https://github.com/Denistanb/Fashion-Mnist)

This project explores the Fashion-MNIST dataset using deep learning techniques. It involves data preprocessing, exploratory data analysis (EDA), and building classification models using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) to accurately classify fashion items.

---

## 📘 Problem Statement

The objective is to classify grayscale images of fashion items into one of 10 categories. Each image is 28x28 pixels, representing items such as T-shirts, trousers, and sneakers. The challenge lies in achieving high classification accuracy using deep learning models.

---

## 📂 Project Structure

- `Fashion Mnist.ipynb` — Jupyter Notebook containing the complete workflow: data loading, EDA, model building, training, and evaluation.
- `fashion-mnist_train.csv` — CSV file containing the training dataset.
- `fashion-mnist_test.csv` — CSV file containing the test dataset.
- `train-images-idx3-ubyte` — Binary file containing training images.
- `train-labels-idx1-ubyte` — Binary file containing training labels.
- `t10k-images-idx3-ubyte` — Binary file containing test images.
- `t10k-labels-idx1-ubyte` — Binary file containing test labels.

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized sample images from the dataset to understand the data distribution.
- Checked the balance of classes to ensure uniform representation.
- Normalized pixel values to enhance model performance.

---

## 🧠 Model Development

### 1️⃣ Artificial Neural Network (ANN)

- **Architecture**: Flatten layer followed by two dense layers with ReLU activation, and an output layer with sigmoid activation.
- **Training**: Used stochastic gradient descent (SGD) optimizer and sparse categorical crossentropy loss.
- **Performance**: Achieved moderate accuracy, serving as a baseline model.

### 2️⃣ Convolutional Neural Network (CNN)

- **Architecture**: Two convolutional layers with ReLU activation and max pooling, followed by a dense layer and an output layer with softmax activation.
- **Training**: Utilized Adam optimizer and sparse categorical crossentropy loss.
- **Performance**: Significantly improved accuracy over the ANN model.

---

## 📈 Evaluation

- Employed classification reports to assess precision, recall, and F1-score.
- Generated confusion matrices to visualize model performance across different classes.

---

## 🧰 Technologies Used

- Python 🐍
- TensorFlow & Keras 🤖
- NumPy & Pandas 🧮
- Matplotlib & Seaborn 📊
- Jupyter Notebook 📓

---

## ▶️ How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Denistanb/Fashion-Mnist.git
   cd Fashion-Mnist
2. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib seaborn tensorflow
3. **Launch the Notebook**:
   ```bash
   jupyter notebook "Fashion Mnist.ipynb"
