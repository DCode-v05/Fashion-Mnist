# Fashion-MNIST Classification

## Project Description
This project focuses on classifying fashion items using the Fashion-MNIST dataset. The goal is to leverage deep learning techniques to accurately categorize grayscale images of clothing and accessories into one of ten predefined classes. The project covers data preprocessing, exploratory data analysis (EDA), and the development of classification models using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

---

## Project Details

### Problem Statement
The objective is to classify 28x28 pixel grayscale images of fashion items into ten categories, such as T-shirts, trousers, and sneakers. The challenge is to achieve high classification accuracy using robust deep learning models.

### Exploratory Data Analysis (EDA)
- Visualized sample images to understand data distribution.
- Checked class balance to ensure uniform representation.
- Normalized pixel values to improve model performance.

### Model Development
#### Artificial Neural Network (ANN)
- Architecture: Flatten layer, two dense layers with ReLU activation, and an output layer with sigmoid activation.
- Training: Stochastic Gradient Descent (SGD) optimizer and sparse categorical crossentropy loss.
- Performance: Served as a baseline model with moderate accuracy.

#### Convolutional Neural Network (CNN)
- Architecture: Two convolutional layers with ReLU activation and max pooling, followed by a dense layer and an output layer with softmax activation.
- Training: Adam optimizer and sparse categorical crossentropy loss.
- Performance: Achieved significantly higher accuracy compared to the ANN model.

### Evaluation
- Used classification reports to assess precision, recall, and F1-score.
- Generated confusion matrices to visualize model performance across classes.

---

## Tech Stack
- Python
- TensorFlow & Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Jupyter Notebook

---

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DCode-v05/Fashion-Mnist.git
   cd Fashion-Mnist
   ```
2. **Install Dependencies:**
   ```bash
   pip install numpy matplotlib seaborn tensorflow
   ```

---

## Usage
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "Fashion Mnist.ipynb"
   ```
2. Open the notebook and follow the cells to run data loading, EDA, model training, and evaluation steps.

---

## Project Structure
- `Fashion Mnist.ipynb` — Jupyter Notebook containing the complete workflow: data loading, EDA, model building, training, and evaluation.
- `fashion-mnist_train.csv` — CSV file containing the training dataset.
- `fashion-mnist_test.csv` — CSV file containing the test dataset.
- `train-images-idx3-ubyte` — Binary file containing training images.
- `train-labels-idx1-ubyte` — Binary file containing training labels.
- `t10k-images-idx3-ubyte` — Binary file containing test images.
- `t10k-labels-idx1-ubyte` — Binary file containing test labels.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.
   
---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
