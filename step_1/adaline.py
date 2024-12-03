import numpy as np

def adaline_train(X, y, learning_rate, epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features) 
    bias = 0 

    for _ in range(epochs):
        linear_output = np.dot(X, weights) + bias
        errors = y - linear_output

        weights += learning_rate * np.dot(errors, X) / n_samples
        bias += learning_rate * np.sum(errors) / n_samples

    return weights, bias

def adaline_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    predictions = np.where(linear_output >= 0, 1, -1)
    return predictions
