import numpy as np
from functions import step_function

def perceptron_train(data, labels, learning_rate=0.01, epochs=1000):
    """Treina o modelo perceptron."""
    n_samples, n_features = data.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for i in range(n_samples):
            linear_output = np.dot(data[i], weights) + bias
            prediction = step_function(linear_output)
            update = learning_rate * (labels[i] - prediction)
            weights += (update * data[i])/2
            bias += update
    return weights, bias

def perceptron_predict(data, weights, bias):
    """Realiza predições com o modelo perceptron."""
    linear_output = np.dot(data, weights) + bias
    return step_function(linear_output)