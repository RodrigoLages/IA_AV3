import numpy as np

def perceptron_train(data, labels, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    n_samples, n_features = data.shape
    n_classes = labels.shape[1]  # Number of classes
    weights = np.zeros((n_features + 1, n_classes))  # Weight matrix for multi-class
    data = np.hstack((np.ones((n_samples, 1)), data))  # Add bias term
    no_improve_count = 0

    mse_history = []

    for epoch in range(epochs):
        # Linear output for all classes
        linear_output = np.dot(data, weights)
        predictions = softmax(linear_output)  # Multi-class prediction using softmax

        # Calculate errors and MSE
        errors = labels - predictions
        mse = np.mean(errors ** 2)
        mse_history.append(mse)

        # Update weights
        weight_update = learning_rate * np.dot(data.T, errors) / n_samples
        weights += weight_update

        # Stopping criteria based on weight update norm
        if np.linalg.norm(weight_update) < tolerance:
            print(f"Convergence reached at epoch {epoch + 1}")
            break

        # Early stopping based on error improvement
        if mse == 0:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping activated due to no improvement.")
                break
        else:
            no_improve_count = 0

    return weights, mse_history


def perceptron_predict(data, weights):
    n_samples = data.shape[0]
    data = np.hstack((np.ones((n_samples, 1)), data))  # Add bias term
    linear_output = np.dot(data, weights)
    predictions = softmax(linear_output)
    return np.argmax(predictions, axis=1)  # Return the class with the highest score


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)