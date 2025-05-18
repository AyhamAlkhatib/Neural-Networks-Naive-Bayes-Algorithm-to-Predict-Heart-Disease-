# Heart Disease Prediction using Naive Bayes and Neural Networks

This project implements two supervised learning models — **Gaussian Naive Bayes** and a **Feedforward Neural Network** — from scratch in Python to predict heart disease using real patient data. It demonstrates a comparison of these algorithms based on multiple evaluation metrics.

---

## 🩺 Data Set

- **Source**: [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Size**: 297 records
- **Features** (13 total):
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target**: Binary variable indicating presence (1) or absence (0) of heart disease

---

## 🛠 Tools and Frameworks

- **Programming Language**: Python
- **Libraries Used**:
  - `numpy` for numerical computation
  - `pandas` for data manipulation
  - `matplotlib` for visualization
- **Platform**: Developed and tested in a local Python environment (no scikit-learn or TensorFlow used)

---

## 🔁 Project Workflow

1. **Data Preprocessing**:
   - Missing values dropped
   - Normalized using min-max scaling
   - 70/30 manual train-test split

2. **Model Implementation**:
   - Gaussian Naive Bayes: manual calculation of priors and conditional probabilities
   - Neural Network: 1 hidden layer (6 neurons), sigmoid activation, backpropagation with MSE loss

3. **Evaluation**:
   - Accuracy, Precision, Recall, and F1-score on test data
   - Learning curves tracked for neural network

---

## 📊 Results (Key Insights)

| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Naive Bayes  | 82.2%    | 78.9%     | 78.9%  | 78.9%    |
| Neural Net   | 82.2%    | 78.9%     | 78.9%  | 78.9%    |

- Both models performed comparably
- Naive Bayes is faster and interpretable
- Neural Network generalizes slightly better with smooth learning dynamics

---

## 📈 Visualizations

- **Training vs Test Accuracy over Epochs** (Neural Network)
- **Training vs Test Loss over Epochs** (Neural Network)

> Note: Plots are included in the PDF report and generated using `matplotlib`.

---

## 📄 Report

For a detailed breakdown of methodology, implementation, and evaluation:
📎 [Final Project Report (PDF)](./final_project_cleaned.pdf)

---
