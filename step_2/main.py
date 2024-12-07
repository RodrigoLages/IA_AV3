import numpy as np
from functions import load_data, normalize, plot_learning_curves, update_results, build_summary, plot_conf_matrices
from perceptron import perceptron_train, perceptron_predict
from adaline import adaline_train, adaline_predict
from mlp import mlp_train, mlp_predict

# Hiperparâmetros
plot = True
R=50
epochs=200
learning_rate=0.01
tolerance = 1e-4
patience = 10
hidden_layers = [50, 50, 50]
last_layer = 20
img_size = 50

# Modelos
usePerceptron = False
useAdaline = False
useMlp = True

# Carregar e normalizar os dados
# Num classes : 20
# data : 640 x 2500
# labels : 640 x 20 | One hot encoded
data, labels = load_data(img_size)
data = normalize(data)

# Validação por Monte Carlo.
n_samples = data.shape[0]
metricsPerceptron = {"accuracy": [], "sensitivity": [], "specificity": []}
metricsAdaline = {"accuracy": [], "sensitivity": [], "specificity": []}
metricsMlp = {"accuracy": [], "sensitivity": [], "specificity": []}
resultsPerceptron = []
resultsAdaline = []
resultsMlp = []

for _ in range(R):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    train_size = int(0.8 * n_samples)
    X_train, X_test = data[:train_size], data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    # Modelo Perceptron
    if usePerceptron:
        weights, mse_history = perceptron_train(X_train, y_train, learning_rate, epochs, tolerance, patience)
        predictions = perceptron_predict(X_test, weights)
        update_results(metricsPerceptron, resultsPerceptron, predictions, y_test, mse_history)

    # Modelo Adaline
    if useAdaline:
        weights, mse_history = adaline_train(X_train, y_train, learning_rate, epochs, tolerance, patience)
        predictions = adaline_predict(X_test, weights)
        update_results(metricsAdaline, resultsAdaline, predictions, y_test, mse_history)

    # Modelo MLP
    if useMlp:
        weights, mse_history = mlp_train(X_train, y_train, hidden_layers, last_layer, learning_rate, epochs, tolerance)
        predictions = mlp_predict(X_test, weights)
        update_results(metricsMlp, resultsMlp, predictions, y_test, mse_history)


    if (_+1) % (R/10) == 0:
        print(f"Rodada: {_+1}")


# Mostar resultados
if usePerceptron:
    bestPerceptron = max(resultsPerceptron, key=lambda x: x['accuracy'])
    worstPerceptron = min(resultsPerceptron, key=lambda x: x['accuracy'])
    summaryPerceptron = build_summary(metricsPerceptron)

    print("\nResumo das métricas do Perceptron:")
    print(summaryPerceptron)
    if (plot):
        plot_conf_matrices(bestPerceptron["conf_matrix"], worstPerceptron["conf_matrix"], "Perceptron")
        plot_learning_curves(bestPerceptron["mse"], worstPerceptron["mse"], "Perceptron")


if useAdaline:
    bestAdaline = max(resultsAdaline, key=lambda x: x['accuracy'])
    worstAdaline = min(resultsAdaline, key=lambda x: x['accuracy'])
    summaryAdaline = build_summary(metricsAdaline)

    print("\nResumo das métricas do Adaline:")
    print(summaryAdaline)
    if (plot):
        plot_conf_matrices(bestAdaline["conf_matrix"], worstAdaline["conf_matrix"], "Adaline")
        plot_learning_curves(bestAdaline["mse"], worstAdaline["mse"], "Adaline")
    

if useMlp:
    bestMlp = max(resultsMlp, key=lambda x: x['accuracy'])
    worstMlp = min(resultsMlp, key=lambda x: x['accuracy'])
    summaryMlp = build_summary(metricsMlp)

    print("\nResumo das métricas do MLP:")
    print(summaryMlp)
    if (plot):
        plot_conf_matrices(bestMlp["conf_matrix"], worstMlp["conf_matrix"], "MLP")
        plot_learning_curves(bestMlp["mse"], bestMlp["mse"], "MLP")

        
    
