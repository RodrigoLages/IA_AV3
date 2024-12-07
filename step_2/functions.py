import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

def load_data(img_size=50):
    pasta_raiz = "RecFac"
    caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
    caminho_pessoas.pop(0)

    C = 20 #Esse é o total de classes 
    X = np.empty((img_size*img_size,0)) # Essa variável X será a matriz de dados de dimensões p x N. 
    Y = np.empty((C,0))
    for i,pessoa in enumerate(caminho_pessoas):
        imagens_pessoa = os.listdir(pessoa)
        for imagens in imagens_pessoa:

            caminho_imagem = os.path.join(pessoa,imagens)
            imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
            imagem_redimensionada = cv2.resize(imagem_original,(img_size,img_size))

            #A imagem pode ser visualizada com esse comando.
            # cv2.imshow("eita",imagem_redimensionada)
            # cv2.waitKey(0)

            #vetorizando a imagem:
            x = imagem_redimensionada.flatten()

            #Empilhando amostra para criar a matriz X que terá dimensão p x N
            X = np.concatenate((
                X,
                x.reshape(img_size*img_size,1)
            ),axis=1)
            

            #one-hot-encoding (A DESENVOLVER)
            y = -np.ones((C,1))
            y[i,0] = 1

            Y = np.concatenate((
                Y,
                y
            ),axis=1)
    return X.T, Y.T

def step_function(x):
    return np.where(x >= 0, 1, -1)

def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    normalized_data = (data - min_val) / (max_val - min_val)
    
    normalized_data = 2 * normalized_data - 1
    
    return normalized_data


def calculate_confusion_matrix(predictions, y_test, n_classes):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(y_test, predictions):
        conf_matrix[true_label, predicted_label] += 1
    return conf_matrix


def calculate_metrics(predictions, y_test, n_classes):
    accuracy = np.mean(predictions == y_test)

    # Custom confusion matrix
    conf_matrix = calculate_confusion_matrix(predictions, y_test, n_classes)
    
    # Sensitivity (Recall) and Specificity
    sensitivity = []
    specificity = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fn + fp)

        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    # Average metrics across classes
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)

    return accuracy, avg_sensitivity, avg_specificity, conf_matrix


def update_results(metrics, results, predictions, y_test, mse_history, n_classes=20):
    accuracy, sensitivity, specificity, conf_matrix = calculate_metrics(predictions, np.argmax(y_test, axis=1), n_classes)

    metrics["accuracy"].append(accuracy)
    metrics["sensitivity"].append(sensitivity)  # Average sensitivity
    metrics["specificity"].append(specificity)  # Average specificity

    results.append({
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "mse": mse_history
    })

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

def plot_conf_matrices(conf_matrix1, conf_matrix2, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["an2i", "at33", "bol", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi", "kawamura", "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"]
    
    # Melhor modelo
    sns.heatmap(conf_matrix1, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"Melhor {model_name}")
    axes[0].set_xlabel("Previsão")
    axes[0].set_ylabel("Valor Real")
    
    # Pior modelo
    sns.heatmap(conf_matrix2, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, ax=axes[1])
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