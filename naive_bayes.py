import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart_cleveland_upload.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Manual 70/30 split
split = int(0.7 * len(df))
train_df = df[:split]
test_df = df[split:]

features = [col for col in df.columns if col != "condition"]

def calculate_prior(df, label_col):
    priors = {}
    classes = sorted(df[label_col].unique())
    for c in classes:
        priors[c] = len(df[df[label_col] == c]) / len(df)
    return priors

def calculate_likelihood(df, feature, value, label_col, label):
    sub = df[df[label_col] == label]
    mean = sub[feature].mean()
    std = sub[feature].std()
    std = std if std > 0 else 1e-6
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean)**2) / (2 * std**2))

def predict_naive_bayes(train_df, X, label_col):
    priors = calculate_prior(train_df, label_col)
    classes = sorted(train_df[label_col].unique())
    preds = []
    probs = []

    for x in X:
        posteriors = {}
        for c in classes:
            likelihood = 1
            for i, feature in enumerate(features):
                likelihood *= calculate_likelihood(train_df, feature, x[i], label_col, c)
            posteriors[c] = priors[c] * likelihood
        pred = max(posteriors, key=posteriors.get)
        preds.append(pred)
        probs.append(posteriors)
    return np.array(preds), probs

def compute_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_metrics(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / len(y_true)
    return acc, precision, recall, f1

train_X = train_df[features].values
train_y = train_df["condition"].values
test_X = test_df[features].values
test_y = test_df["condition"].values

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for i in range(1, 11):  # Simulate "iterations" by increasing training subset size
    size = int(len(train_X) * i / 10)
    temp_train_df = train_df[:size]
    
    train_preds, _ = predict_naive_bayes(temp_train_df, temp_train_df[features].values, "condition")
    test_preds, _ = predict_naive_bayes(temp_train_df, test_X, "condition")

    train_loss = compute_loss(temp_train_df["condition"].values, train_preds)
    test_loss = compute_loss(test_y, test_preds)

    train_acc = compute_accuracy(temp_train_df["condition"].values, train_preds)
    test_acc = compute_accuracy(test_y, test_preds)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

final_preds, _ = predict_naive_bayes(train_df, test_X, "condition")
accuracy, precision, recall, f1 = compute_metrics(test_y, final_preds)

print("\n=== Final Evaluation on Test Set ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")