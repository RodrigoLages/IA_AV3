import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(plot: False):
    csvData = pd.read_csv('spiral.csv', header=None)

    csvData.columns = ['X', 'Y', 'Group']

    if plot:
        groups = csvData.groupby('Group')
        plt.figure(figsize=(8, 6))

        for name, group in groups:
            plt.scatter(group['X'], group['Y'], label=f'Group {name}', alpha=0.7)

        plt.title('Spiral Data Scatter Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    data = csvData[['X', 'Y']].to_numpy()  # Shape: 2 x N
    labels = csvData['Group'].to_numpy() # Shape: N
    return data, labels

def step_function(x):
    return np.where(x >= 0, 1, -1)

def normalize(data):
    """Normaliza os dados para o intervalo [-1, 1]."""
    return (data - data.min()) / (data.max() - data.min()) * 2 - 1


def calculate_metrics(predictions, y_test):
    """Calcula métricas de desempenho para os modelos."""
    accuracy = np.mean(predictions == y_test)
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == -1) & (y_test == -1))
    fp = np.sum((predictions == 1) & (y_test == -1))
    fn = np.sum((predictions == -1) & (y_test == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return accuracy, sensitivity, specificity

def update_results(metrics, results, predictions, y_test, mse_history):
    """Atualiza as métricas e resultados para um modelo."""
    accuracy, sensitivity, specificity = calculate_metrics(predictions, y_test)

    metrics["accuracy"].append(accuracy)
    metrics["sensitivity"].append(sensitivity)
    metrics["specificity"].append(specificity)

    results.append({"accuracy": accuracy, "conf_matrix": conf_matrix(y_test, predictions), "mse": mse_history})

def build_summary(metrics):
    """Constrói um DataFrame com o resumo das métricas."""
    return pd.DataFrame({
        "Metric": ["Accuracy", "Sensitivity", "Specificity"],
        "Mean": [
            np.mean(metrics["accuracy"]),
            np.mean(metrics["sensitivity"]),
            np.mean(metrics["specificity"]),
        ],
        "StdDev": [
            np.std(metrics["accuracy"]),
            np.std(metrics["sensitivity"]),
            np.std(metrics["specificity"]),
        ],
        "Max": [
            np.max(metrics["accuracy"]),
            np.max(metrics["sensitivity"]),
            np.max(metrics["specificity"]),
        ],
        "Min": [
            np.min(metrics["accuracy"]),
            np.min(metrics["sensitivity"]),
            np.min(metrics["specificity"]),
        ],
    })

def conf_matrix(y_true, y_pred):
    """Calcula a matriz de confusão."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == -1) & (y_true == -1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    
    return np.array([[tn, fp], [fn, tp]])

def plot_conf_matrices(conf_matrix1, conf_matrix2, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Melhor modelo
    sns.heatmap(conf_matrix1, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred -1", "Pred 1"], yticklabels=["True -1", "True 1"], ax=axes[0])
    axes[0].set_title(f"Melhor {model_name}")
    axes[0].set_xlabel("Previsão")
    axes[0].set_ylabel("Valor Real")
    
    # Pior modelo
    sns.heatmap(conf_matrix2, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred -1", "Pred 1"], yticklabels=["True -1", "True 1"], ax=axes[1])
    axes[1].set_title(f"Pior {model_name}")
    axes[1].set_xlabel("Previsão")
    axes[1].set_ylabel("Valor Real")
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(best_case_mse, worst_case_mse, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Melhor caso
    axs[0].plot(range(1, len(best_case_mse) + 1), best_case_mse, marker='o', linestyle='-', color='g')
    axs[0].set_title(f"{title} - Melhor Caso")
    axs[0].set_xlabel("Épocas")
    axs[0].set_ylabel("Erro Quadrático Médio (MSE)")
    axs[0].grid(True)

    # Pior caso
    axs[1].plot(range(1, len(worst_case_mse) + 1), worst_case_mse, marker='o', linestyle='-', color='r')
    axs[1].set_title(f"{title} - Pior Caso")
    axs[1].set_xlabel("Épocas")
    axs[1].grid(True)

    # Ajustar layout
    plt.tight_layout()
    plt.show()