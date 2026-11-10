# Classifying-Heartbeats-from-ECG-Signals
# Classifying-Heartbeats-from-ECG-Signals

## 📌 Project Overview

This project focuses on classifying ECG (Electrocardiogram) heartbeat signals into five different categories using a Fully Connected Neural Network built from scratch with PyTorch.

The main objective is to identify different types of heartbeats and understand how machine learning can be applied to medical signal classification.

The project uses the **MIT-BIH Arrhythmia Dataset**, which contains ECG heartbeat signals belonging to five classes.

## 🎯 Objectives

- Build a fully connected neural network using PyTorch.
- Implement He weight initialization.
- Implement Softmax Cross-Entropy loss.
- Perform backpropagation and gradient checking.
- Implement mini-batch gradient descent with momentum.
- Apply weight decay for regularization.
- Compare different learning rates.
- Evaluate the model using accuracy, balanced accuracy, and a confusion matrix.
- Develop an interactive ECG prediction application using Streamlit.

## 📊 Dataset Description

**Dataset:** MIT-BIH Arrhythmia Dataset

Each heartbeat contains 187 ECG signal values and one class label.

| Class | Heartbeat Type |
|---|---|
| N | Normal |
| S | Supraventricular Ectopic |
| V | Ventricular Ectopic |
| F | Fusion |
| Q | Unknown / Paced |

### Dataset Distribution

| Class | Training Samples |
|---|---:|
| N | 72,471 |
| S | 2,223 |
| V | 5,788 |
| F | 641 |
| Q | 6,431 |
| **Total** | **87,554** |

The dataset is imbalanced because the Normal class contains significantly more samples than the other classes.

## 🧠 Neural Network Architecture

The project uses a fully connected neural network with the following architecture:

```text
Input Layer: 187 Neurons
        |
        v
Hidden Layer 1: 128 Neurons
Activation: ReLU
        |
        v
Hidden Layer 2: 64 Neurons
Activation: ReLU
        |
        v
Output Layer: 5 Neurons
        |
        v
Predicted Heartbeat Class
```

**Total Trainable Parameters: 32,645**

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Input Features | 187 |
| Hidden Layer 1 | 128 |
| Hidden Layer 2 | 64 |
| Output Classes | 5 |
| Batch Size | 128 |
| Epochs | 30 |
| Momentum | 0.9 |
| Weight Decay | 0.0001 |
| Learning Rates | 0.01, 0.1, 0.5 |
| Weight Initialization | He Initialization |
| Activation Function | ReLU |
| Loss Function | Softmax Cross-Entropy |

The model is trained using mini-batch gradient descent with momentum. Three learning rates are compared, and the best model is selected using validation loss.

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Navigate to the project directory

```bash
cd ECG-Heartbeat-Classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the following files inside the `data` folder:

- `mitbih_train.csv`
- `mitbih_test.csv`

### 5. Prepare the data

```bash
python src/prepare.py
```

### 6. Perform gradient checking

```bash
python src/grad_check.py
```

### 7. Train the neural network

```bash
python src/train.py
```

### 8. Evaluate the model

```bash
python src/test_model.py
```

### 9. Run the Streamlit application

```bash
streamlit run app/app.py
```

## 📈 Model Evaluation

The model is evaluated using the following performance metrics:

- Overall Test Accuracy
- Balanced Accuracy
- Per-Class Accuracy
- Confusion Matrix

### Results

| Evaluation Metric | Result |
|---|---|
| Test Accuracy | Add actual result |
| Balanced Accuracy | Add actual result |
| Best Learning Rate | Add selected value |
| Best Epoch | Add selected epoch |

### Visual Results

**1. ECG Heartbeat Examples**

Add your `example_beats.png` here.

**2. Learning Rate Comparison**

Add your `learning_rate_comparison.png` here.

**3. Training and Validation Loss**

Add your `loss_curves.png` here.

**4. Training and Validation Accuracy**

Add your `accuracy_curves.png` here.

**5. Confusion Matrix**

Add your `confusion_matrix.png` here.

## 🔍 Key Analysis

The project investigates six important questions:

1. How does an always-N classifier perform compared with the trained neural network?
2. Which heartbeat class has the lowest classification accuracy?
3. Does the training process demonstrate overfitting or underfitting?
4. Why is He initialization important for ReLU networks?
5. How do momentum and weight decay affect neural network training?
6. How can oversampling or class-weighted loss improve performance on rare heartbeat classes?

## 💡 Future Improvements

- Improve classification of minority heartbeat categories.
- Implement oversampling and class-weighted loss.
- Experiment with deeper neural network architectures.
- Explore CNN-based ECG classification.
- Improve the Streamlit interface.
- Investigate real-time ECG signal classification.

## 🏁 Conclusion

This project demonstrates the practical implementation of a neural network for ECG heartbeat classification.

By implementing the network and optimization process, we gained practical experience in neural networks, backpropagation, gradient descent, regularization, and model evaluation.

The project also highlights the importance of handling imbalanced datasets when developing machine learning solutions for medical applications.

## 👨‍💻 Author

Omar Masood Ahsan

Electrical Engineering | Machine Learning | Embedded Systems

## ⭐ Acknowledgements

- MIT-BIH Arrhythmia Dataset
- PyTorch Documentation
- NumPy and Pandas Documentation
- Streamlit Documentation

---

*This project is developed for academic and educational purposes. It is not intended to replace professional medical diagnosis.*
