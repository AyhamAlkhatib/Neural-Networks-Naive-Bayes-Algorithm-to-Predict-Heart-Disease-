import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size)
    w2 = np.random.randn(hidden_size, output_size)
    return w1, w2

def forward_pass(x, w1, w2):
    z1 = x @ w1
    a1 = sigmoid(z1)
    z2 = a1 @ w2
    a2 = sigmoid(z2)
    return a1, a2

def backward_pass(x, y, a1, a2, w1, w2, lr):
    d2 = (a2 - y) * sigmoid_derivative(a2)
    d1 = (d2 @ w2.T) * sigmoid_derivative(a1)
    w2 -= a1.T @ d2 * lr
    w1 -= x.T @ d1 * lr
    return w1, w2

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def compute_accuracy(y_pred, y_true):
    pred_labels = (y_pred >= 0.5).astype(int)
    return np.mean(pred_labels == y_true)

def train_neural_net(x_train, y_train, x_test, y_test, hidden_size=8, lr=0.1, epochs=100):
    input_size = x_train.shape[1]
    output_size = 1
    w1, w2 = initialize_weights(input_size, hidden_size, output_size)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        a1, a2 = forward_pass(x_train, w1, w2)
        w1, w2 = backward_pass(x_train, y_train, a1, a2, w1, w2, lr)

        train_loss = compute_loss(a2, y_train)
        train_acc = compute_accuracy(a2, y_train)
        a1_test, a2_test = forward_pass(x_test, w1, w2)
        test_loss = compute_loss(a2_test, y_test)
        test_acc = compute_accuracy(a2_test, y_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return w1, w2, train_losses, test_losses, train_accuracies, test_accuracies

def main():
    df = pd.read_csv("heart_cleveland_upload.csv")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    x = df.drop(columns="condition").values
    y = df["condition"].values.reshape(-1, 1)

    x = (x - np.min(x, axis=0)) / (np.ptp(x, axis=0) + 1e-8)

    split = int(0.7 * len(df))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    w1, w2, train_losses, test_losses, train_accs, test_accs = train_neural_net(x_train, y_train, x_test, y_test)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Final metrics
    _, y_pred = forward_pass(x_test, w1, w2)
    y_pred = (y_pred >= 0.5).astype(int)
    y_true = y_test

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = np.mean(y_pred == y_true)

    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    main()