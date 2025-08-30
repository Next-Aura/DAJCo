import numpy as np         # numpy for numerical operations
from typing import Literal # for type hinting of string literals

class BasicRegressor:
    """
    Gradient Supported Basic Regressor (GSBR) for linear regression with optional regularization.
    Implements gradient descent to minimize mean squared error (MSE) or root mean squared error (RMSE)
    with L1, L2, or ElasticNet penalties.
    """
    
    def __init__(self, max_iter: int=100, learning_rate: float=0.01, verbose: int=0, 
                 penalty: Literal["l1", "l2", "elasticnet"] | None="l2", 
                 alpha: float=0.0001, l1_ratio: float=0.5, 
                 fit_intercept: bool=True, tol: float=0.0001, 
                 loss: Literal["mse", "rmse"] | None="mse"):
        """
        Initialize the regressor with hyperparameters.

        Args:
            max_iter: Maximum number of gradient descent iterations.
            learning_rate: Step size for gradient descent updates.
            verbose: If 1, print training progress (epoch, weights, bias, loss).
            penalty: Regularization type ('l1', 'l2', 'elasticnet', or None).
            alpha: Regularization strength.
            l1_ratio: Mixing parameter for ElasticNet (0 = L2, 1 = L1).
            fit_intercept: If True, include a bias term (intercept).
            tol: Tolerance for early stopping based on loss convergence.
            loss: Loss function to minimize ('mse' or 'rmse').
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.loss = loss

        self.loss_history = []  # Track loss at each iteration
        self.weights = None     # Model weights (to be initialized during fit)
        self.b = 0.0           # Bias term (intercept)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss function (MSE or RMSE) with optional regularization penalty.

        Args:
            X: Input features (n_samples, n_features).
            y: Target values (n_samples,).

        Returns:
            float: Loss value (MSE or RMSE) plus regularization penalty.
        """
        # Calculate prediction error: y_pred = Xw + b
        error = X @ self.weights + self.b - y
        mse = np.mean(error**2)

        # Add regularization penalty
        penalty = 0.0
        if self.penalty == "l1":
            # L1 penalty: alpha * ||w||_1
            penalty = self.alpha * np.sum(np.abs(self.weights))
        elif self.penalty == "l2":
            # L2 penalty: alpha * ||w||_2^2
            penalty = self.alpha * np.sum(self.weights**2)
        elif self.penalty == "elasticnet":
            # ElasticNet: alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_2^2)
            l1 = self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
            penalty = self.alpha * (l1 + l2)

        # Return RMSE if specified, otherwise MSE
        return np.sqrt(mse) if self.loss == "rmse" else mse + penalty

    def grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute gradients of the loss function with respect to weights and bias.

        Args:
            X: Input features (n_samples, n_features).
            y: Target values (n_samples,).

        Returns:
            tuple: Gradients for weights (np.ndarray) and bias (float).
        """
        # Prediction error: f = Xw + b - y
        f = X @ self.weights + self.b - y

        # Gradient w.r.t. weights: (2/n) * X^T * error
        grad_w = X.T @ (2 * f) / len(X)

        # Gradient w.r.t. bias: (2/n) * mean(error) if intercept is fitted
        grad_b = np.mean(2 * f) if self.fit_intercept else 0.0

        # Add regularization gradient
        grad_w_penalty = np.zeros_like(self.weights)
        if self.penalty == "l1":
            # L1 gradient: alpha * sign(w)
            grad_w_penalty = self.alpha * np.sign(self.weights)
        elif self.penalty == "l2":
            # L2 gradient: 2 * alpha * w
            grad_w_penalty = 2 * self.alpha * self.weights
        elif self.penalty == "elasticnet":
            # ElasticNet gradient: alpha * (l1_ratio * sign(w) + 2 * (1 - l1_ratio) * w)
            l1 = self.l1_ratio * np.sign(self.weights)
            l2 = 2 * (1 - self.l1_ratio) * self.weights
            grad_w_penalty = self.alpha * (l1 + l2)

        grad_w += grad_w_penalty
        return grad_w, grad_b

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict target values for test data.

        Args:
            X_test: Test features (n_samples, n_features or n_features,).

        Returns:
            np.ndarray: Predicted values.
        """
        # Handle 1D input by reshaping to (n_samples, 1)
        X_processed = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test

        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Compute predictions: y_pred = Xw + b
        return X_processed @ self.weights + self.b

    def fit(self, X_train, y_train):
        """
        Train the model using gradient descent.

        Args:
            X_train: Training features (n_samples, n_features) - can be dense or sparse matrix.
            y_train: Training target values (n_samples,).
        """
        # Handle different input types (dense arrays, sparse matrices)
        from scipy.sparse import issparse
        
        if issparse(X_train):
            # Handle sparse matrices
            X_processed = X_train.toarray()
        else:
            # Handle dense arrays
            X_processed = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train

        # Convert y_train to numpy array if not already
        y_processed = np.asarray(y_train)

        # Ensure data is numeric and handle type conversion
        try:
            X_processed = X_processed.astype(np.float64)
            y_processed = y_processed.astype(np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Input data must be numeric. Found non-numeric types in training data. "
                f"Error: {str(e)}"
            )

        # Validate input data for NaN/Inf values
        if not np.all(np.isfinite(X_processed)):
            raise ValueError("X_train contains NaN or Infinity values.")
        if not np.all(np.isfinite(y_processed)):
            raise ValueError("y_train contains NaN or Infinity values.")
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"X_train samples ({X_processed.shape[0]}) must match "
                f"y_train samples ({y_processed.shape[0]})."
            )

        # Initialize weights if None or mismatched shape
        num_samples, num_features = X_processed.shape
        if self.weights is None or self.weights.shape[0] != num_features:
            self.weights = np.zeros(num_features)

        # Gradient descent loop
        for i in range(self.max_iter):
            # Compute gradients
            grad_w, grad_b = self.grad(X_processed, y_processed)

            # Update weights and bias
            self.weights -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.b -= self.learning_rate * grad_b

            # Compute and store loss
            mse = self._loss(X_processed, y_processed)
            self.loss_history.append(mse)

            # Check for numerical stability
            if not np.all(np.isfinite(self.weights)) or (self.fit_intercept and not np.isfinite(self.b)):
                print(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training.")
                break

            # Print progress if verbose
            if self.verbose == 1:
                print(f"Epoch {i + 1}: Avg weight: {np.mean(self.weights):.4f}, "
                      f"Bias: {self.b:.4f}, Loss: {mse:.4f}")

            # Check for convergence (skip first iteration)
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break

    def loss_score(self):
        """
        Get the mean loss over all iterations.
        """
        if not self.loss_history:
            raise ValueError("No loss history available. Train the model first.")
        
        mean = np.mean(self.loss_history)
        return mean