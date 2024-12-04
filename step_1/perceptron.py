import numpy as np
from functions import step_function

def perceptron_train(data, labels, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    n_samples, n_features = data.shape
    weights = np.zeros(n_features + 1)
    data = np.hstack((np.ones((n_samples, 1)), data)) 
    no_improve_count = 0

    for epoch in range(epochs):
        previous_weights = weights.copy()
        erro = False

        for i in range(n_samples):
            x_t = data[i]
            linear_output = np.dot(weights, x_t)
            prediction = step_function(linear_output)
            error = labels[i] - prediction

            if error != 0:
                erro = True
                weights += (learning_rate * error * x_t) / 2

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

    return weights

def perceptron_predict(data, weights):
    n_samples = data.shape[0]
    data = np.hstack((np.ones((n_samples, 1)), data)) 
    linear_output = np.dot(data, weights)
    return step_function(linear_output)