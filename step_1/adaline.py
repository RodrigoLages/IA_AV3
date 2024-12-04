import numpy as np

def adaline_train(X, y, learning_rate, epochs, tolerance=1e-4, patience=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)
    X = np.hstack((np.ones((n_samples, 1)), X))

    best_weights = weights.copy()
    no_improve_count = 0
    previous_mse = float('inf')

    for epoch in range(epochs):
        linear_output = np.dot(X, weights)
        errors = y - linear_output

        weights += learning_rate * np.dot(errors, X) / n_samples

        mse = np.mean(errors ** 2)

        # Convergence criterion based on MSE
        if abs(previous_mse - mse) < tolerance:
            print(f"Converged at epoch {epoch + 1}")
            break

        if mse >= previous_mse:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Stopping due to lack of improvement in MSE at epoch {epoch + 1}")
                break
        else:
            no_improve_count = 0
            best_weights = weights.copy()

        previous_mse = mse

    return best_weights

def adaline_predict(X, weights):
    n_samples = X.shape[0]
    X = np.hstack((np.ones((n_samples, 1)), X))
    linear_output = np.dot(X, weights)
    predictions = np.where(linear_output >= 0, 1, -1)
    return predictions
