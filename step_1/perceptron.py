import numpy as np
from functions import step_function

def perceptron_train(data, labels, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    n_samples, n_features = data.shape
    weights = np.zeros(n_features + 1)  # Inclui o bias nos pesos
    data = np.hstack((np.ones((n_samples, 1)), data))  # Adiciona o termo de bias aos dados
    no_improve_count = 0

    mse_history = []

    for epoch in range(epochs):
        # Calcula a saída linear para todas as amostras
        linear_output = np.dot(data, weights)
        predictions = step_function(linear_output)

        # Calcula o erro e o MSE
        errors = labels - predictions
        mse = np.mean(errors ** 2)
        mse_history.append(mse)


        weight_update = learning_rate * np.dot(errors, data) / n_samples
        weights += weight_update

        # Critério de parada baseado na norma da atualização dos pesos
        if np.linalg.norm(weight_update) < tolerance:
            print(f"Convergência alcançada na época {epoch + 1}")
            break

        # Early stopping baseado no erro
        if mse == 0:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping ativado devido a nenhuma melhora.")
                break
        else:
            no_improve_count = 0

    return weights, mse_history

def perceptron_predict(data, weights):
    n_samples = data.shape[0]
    data = np.hstack((np.ones((n_samples, 1)), data)) 
    linear_output = np.dot(data, weights)
    return step_function(linear_output)