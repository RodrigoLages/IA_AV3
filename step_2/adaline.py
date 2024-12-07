import numpy as np

import numpy as np

def adaline_train(X, y, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    n_samples, n_features = X.shape
    n_classes = y.shape[1]  # Número de classes
    weights = np.zeros((n_features + 1, n_classes))  # Matriz de pesos incluindo o bias
    X = np.hstack((np.ones((n_samples, 1)), X))  # Adiciona o termo de bias ao input

    mse_history = []
    no_improve_count = 0

    for epoch in range(epochs):
        # Forward pass: saída linear
        linear_output = np.dot(X, weights)
        predictions = softmax(linear_output)  # Predições multi-classe

        # Calcula os erros e MSE
        errors = y - predictions
        mse = np.mean(errors ** 2)
        mse_history.append(mse)

        # Atualiza os pesos
        weight_update = learning_rate * np.dot(X.T, errors) / n_samples
        weights += weight_update

        # Critério de parada com base na norma da atualização dos pesos
        if np.linalg.norm(weight_update) < tolerance:
            print(f"Convergência alcançada na época {epoch + 1}")
            break

        # Early stopping com base na melhoria do MSE
        if epoch > 0 and mse >= mse_history[-2]:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping ativado devido à falta de melhoria.")
                break
        else:
            no_improve_count = 0

    return weights, mse_history

def adaline_predict(X, weights):
    n_samples = X.shape[0]
    X = np.hstack((np.ones((n_samples, 1)), X))  # Adiciona o termo de bias
    linear_output = np.dot(X, weights)
    predictions = softmax(linear_output)  # Calcula a probabilidade para cada classe
    return np.argmax(predictions, axis=1)  # Retorna a classe com maior probabilidad

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)