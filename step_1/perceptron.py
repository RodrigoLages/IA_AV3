import numpy as np
from functions import step_function

def perceptron_train(data, labels, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    n_samples, n_features = data.shape
    weights = np.zeros(n_features + 1)  # Include bias in the weights vector
    data = np.hstack((np.ones((n_samples, 1)), data))  # Add bias term to data
    no_improve_count = 0

    mse_history = []

    for epoch in range(epochs):
        previous_weights = weights.copy()
        erro = False
        predictions = []

        for i in range(n_samples):
            x_t = data[i]
            linear_output = np.dot(weights, x_t)
            prediction = step_function(linear_output)
            predictions.append(prediction)
            error = labels[i] - prediction

            if error != 0:
                erro = True
                weights += (learning_rate * error * x_t) / 2

        # Calculate MSE for this epoch
        errors = labels - np.array(predictions)
        mse = np.mean(errors ** 2)
        mse_history.append(mse)

        weight_change = np.linalg.norm(weights - previous_weights)
        if weight_change < tolerance:
            break

        # Early stopping
        if not erro:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
        else:
            no_improve_count = 0

    return weights, mse_history

def perceptron_predict(data, weights):
    n_samples = data.shape[0]
    data = np.hstack((np.ones((n_samples, 1)), data)) 
    linear_output = np.dot(data, weights)
    return step_function(linear_output)