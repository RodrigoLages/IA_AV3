import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import load_data, normalize, update_results, build_summary, plot_conf_matrices
from perceptron import perceptron_train, perceptron_predict
from adaline import adaline_train, adaline_predict

data, labels = load_data(plot=False)
data = normalize(data)

# Hiperparâmetros
learning_rate=0.01
epochs=100
R=10
tolerance = 1e-4
patience = 10

# Modelos
usePerceptron = False
useAdaline = True
useMlp = False

# Validação por Monte Carlo.
n_samples = data.shape[0]
metricsPerceptron = {"accuracy": [], "sensitivity": [], "specificity": []}
metricsAdaline = {"accuracy": [], "sensitivity": [], "specificity": []}
metricsMlp = {"accuracy": [], "sensitivity": [], "specificity": []}
resultsPerceptron = []
resultsAdaline = []
resultsMlp = []

# TODO: Implementar a curva de aprendizados para o melhor e pior valor de cada modelo. Vai precisar retornar o MSE no treinamento

for _ in range(R):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    train_size = int(0.8 * n_samples)
    X_train, X_test = data[:train_size], data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    # Modelo Perceptron
    if usePerceptron:
        weights = perceptron_train(X_train, y_train, learning_rate, epochs, tolerance, patience)
        predictions = perceptron_predict(X_test, weights)
        update_results(metricsPerceptron, resultsPerceptron, predictions, y_test)

    # Modelo Adaline
    if useAdaline:
        weights = adaline_train(X_train, y_train, learning_rate, epochs, tolerance, patience)
        predictions = adaline_predict(X_test, weights)
        update_results(metricsAdaline, resultsAdaline, predictions, y_test)

    # Modelo MLP
    if useMlp:
        # Treinar o modelo MLP
        # predictions = modelo_mlp.predict(X_test)
        # update_results(metricsMlp, resultsMlp, predictions, y_test)
        pass


    if (_+1) % (R/10) == 0:
        print(f"Rodada: {_+1}")


# Mostar resultados
if usePerceptron:
    bestPerceptron = max(resultsPerceptron, key=lambda x: x['accuracy'])
    worstPerceptron = min(resultsPerceptron, key=lambda x: x['accuracy'])
    summaryPerceptron = build_summary(metricsPerceptron)

    print("\nResumo das métricas do Perceptron:")
    print(summaryPerceptron)
    plot_conf_matrices(bestPerceptron["conf_matrix"], worstPerceptron["conf_matrix"], "Perceptron")


if useAdaline:
    bestAdaline = max(resultsAdaline, key=lambda x: x['accuracy'])
    worstAdaline = min(resultsAdaline, key=lambda x: x['accuracy'])
    summaryAdaline = build_summary(metricsAdaline)

    print("\nResumo das métricas do Adaline:")
    print(summaryAdaline)
    plot_conf_matrices(bestAdaline["conf_matrix"], worstAdaline["conf_matrix"], "Adaline")

if useMlp:
    bestMlp = max(resultsMlp, key=lambda x: x['accuracy'])
    worstMlp = min(resultsMlp, key=lambda x: x['accuracy'])
    summaryMlp = build_summary(metricsMlp)

    print("\nResumo das métricas do MLP:")
    print(summaryMlp)
    plot_conf_matrices(bestMlp["conf_matrix"], worstMlp["conf_matrix"], "MLP")
    
