import numpy as np

activation_func = lambda x: np.tanh(x)
activation_derivative = lambda a: 1 - a**2

def mlp_train(data, labels, hidden_layers, last_layer=1, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
    # Normalização dos dados de entrada
    data = data.T  # Transposta para formato (features, samples)
    labels = labels.reshape(1, -1)  # Formato (1, samples)
    n_features, n_samples = data.shape

    # Definição das camadas
    layers = [n_features] + hidden_layers + [last_layer]

    # Inicialização dos pesos (incluindo bias) para cada camada
    weights = [
        np.random.randn(layers[i + 1], layers[i] + 1) * np.sqrt(2 / layers[i])
        for i in range(len(layers) - 1)
    ]

    mse_history = []
    no_improve_count = 0

    for epoch in range(epochs):
        # Forward pass
        activations = [data]
        zs = []  # Somatórios (z) em cada camada

        for w in weights:
            # Adiciona o bias à entrada atual
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
            z = np.dot(w, a_with_bias)
            zs.append(z)
            a = activation_func(z)
            activations.append(a)

        # Cálculo do erro e MSE
        y_pred = activations[-1]
        mse = np.mean((y_pred - labels) ** 2)
        mse_history.append(mse)

        # Critério de early stopping baseado no MSE
        if epoch > 0 and mse >= mse_history[-2]:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping ativado na época {epoch + 1}.")
                break
        else:
            no_improve_count = 0

        # Critério de convergência baseado na diferença de MSE
        if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tolerance:
            print(f"Convergência alcançada na época {epoch + 1}.")
            break

        # Backpropagation
        deltas = [None] * len(weights)
        deltas[-1] = (y_pred - labels) * activation_derivative(y_pred)

        for l in range(len(weights) - 2, -1, -1):
            a_with_bias = np.vstack([np.ones((1, activations[l + 1].shape[1])), activations[l + 1]])
            deltas[l] = np.dot(weights[l + 1][:, 1:].T, deltas[l + 1]) * activation_derivative(activations[l + 1])

        for l in range(len(weights)):
            a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]])
            weights[l] -= learning_rate * np.dot(deltas[l], a_with_bias.T) / n_samples

    return weights, mse_history


def mlp_predict(data, weights):
    data = data.T  # Transposta para formato (features, samples)
    activations = data

    for w in weights:
        a_with_bias = np.vstack([np.ones((1, activations.shape[1])), activations])
        z = np.dot(w, a_with_bias)
        activations = activation_func(z)

    final_output = activations.T  # Shape (samples, last_layer)
    predictions = np.where(final_output >= 0, 1, -1)

    return predictions.flatten()