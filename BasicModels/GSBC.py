import numpy as np                          # numpy for numerical operations
from scipy.sparse import issparse, spmatrix # for sparse matrix handling
from typing import Literal                  # for type hinting of specific string literals

class BasicClassifier:
    """
    Gradient Supported Basic Classifier (GSBC) for binary and multi-class classification.
    Uses logistic regression with gradient descent to minimize binary cross-entropy loss.
    Supports One-vs-Rest (OvR) strategy for multi-class classification.
    Handles both dense and sparse input matrices.
    """
    
    def __init__(
        self, 
        max_iter: int=1000, 
        learning_rate: float=0.001, 
        verbose: int=0, 
        fit_intercept: bool=True, 
        tol: float=0.0001,
        penalty: Literal['l1', 'l2', 'elasticnet'] | None=None,
        alpha: float=0.001,
        l1_ratio: float=0.5
        ):
        """
        Initialize the classifier with hyperparameters.

        Args:
            max_iter: Maximum number of gradient descent iterations.
            learning_rate: Step size for gradient descent updates.
            verbose: If 1, print training progress (epoch, loss, etc.).
            fit_intercept: If True, include a bias term (intercept).
            tol: Tolerance for early stopping based on loss convergence.
            penalty: Type of regularization ('l1', 'l2', 'elasticnet') or None.
            alpha: Regularization strength (used if penalty is not None).
            l1_ratio: Mixing parameter for elastic net (0 <= l1_ratio <= 1).

        Raises:
            ValueError: If max_iter, learning_rate, alpha, or tol are non-positive,
                        or if penalty/l1_ratio are invalid.
        """
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if tol <= 0:
            raise ValueError("tol must be positive")

        if penalty not in (None, 'l1', 'l2', 'elasticnet'):
            raise ValueError("penalty must be one of None, 'l1', 'l2', or 'elasticnet'")

        if penalty == 'elasticnet' and not (0 <= l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1 for elasticnet penalty")
        
        if alpha <= 0:
            raise ValueError("alpha must be non-negative")

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.weights = None           # Model weights (initialized during fit)
        self.b = 0.0                  # Bias term (intercept)
        self.loss_history = []        # Track loss at each iteration
        self.classes = None           # Unique class labels
        self.n_classes = 0            # Number of classes
        self.binary_classifiers = []  # List of OvR classifiers for multi-class

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function for logistic regression.

        Args:
            z: Input values (X @ weights + bias).

        Returns:
            np.ndarray: Sigmoid output, probability between 0 and 1.
        """
        # Sigmoid: 1 / (1 + e^(-z)) maps any real number to (0,1)
        return 1 / (1 + np.exp(-z))

    def binary_ce(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss (log loss).

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities from sigmoid.

        Returns:
            float: Mean binary cross-entropy loss.
        """
        # Prevent log(0) by clipping probabilities
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        # Binary cross-entropy: -mean(y * log(p) + (1-y) * log(1-p))
        loss = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        
        # Add regularization penalty if specified
        if self.penalty == 'l1':
            loss += self.alpha * np.sum(np.abs(self.weights))
            # L1 regularization adds the absolute values of weights
            # Formula: alpha * ||w||_1

        elif self.penalty == 'l2':
            loss += self.alpha * np.sum(self.weights**2)
            # L2 regularization adds the squared values of weights
            # Formula: alpha * ||w||_2^2

        elif self.penalty == 'elasticnet':
            l1 = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * self.alpha * np.sum(self.weights**2)
            loss += l1 + l2
            # Elastic Net combines L1 and L2 penalties
            # Formula: alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_2^2)

        return np.mean(loss)

    def grad(self, X: np.ndarray | spmatrix, y_true: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute gradients of the binary cross-entropy loss w.r.t. weights and bias.

        Args:
            X: Input features (n_samples, n_features), dense or sparse.
            y_true: True binary labels (n_samples,).

        Returns:
            tuple: Gradients for weights (np.ndarray) and bias (float).
        """
        # Compute predictions: z = Xw + b, p = sigmoid(z)
        z = X @ self.weights
        if self.fit_intercept:
            z += self.b
        y_pred_proba = self.sigmoid(z)

        # Error: difference between predicted probabilities and true labels
        error = y_pred_proba - y_true

        # Gradient w.r.t. weights: (1/n) * X^T * error
        grad_w = X.T @ error / X.shape[0]
        # Gradient w.r.t. bias: mean(error) if intercept is fitted
        grad_b = np.mean(error) if self.fit_intercept else 0.0

        # Add regularization gradient if specified
        if self.penalty == 'l1':
            grad_w += self.alpha * np.sign(self.weights)
            # L1 regularization gradient: alpha * sign(w)

        elif self.penalty == 'l2':
            grad_w += 2 * self.alpha * self.weights
            # L2 regulation gradient: 2 * alpha * w

        elif self.penalty == 'elasticnet':
            l1 = self.l1_ratio * np.sign(self.weights)
            l2 = 2 * ((1 - self.l1_ratio) * self.weights)
            grad_w += self.alpha * (l1 + l2)
            # Elastic Net gradient: alpha * (l1_ratio * sign(w) + 2 * (1 - l1_ratio) * w)

        return grad_w, grad_b

    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray):
        """
        Train the classifier using gradient descent.

        For binary classification, trains a single logistic regression model.
        For multi-class, trains one binary classifier per class using One-vs-Rest (OvR).

        Args:
            X_train: Training features (n_samples, n_features), dense or sparse.
            y_train: Training target labels (n_samples,).

        Raises:
            ValueError: If input data contains NaN/Inf or mismatched shapes.
        """
        # Preprocess input X
        if not issparse(X_train):
            X_processed = X_train.reshape(-1, 1) if X_train.ndim == 1 else np.asarray(X_train)
        else:
            X_processed = X_train

        num_samples, num_features = X_processed.shape

        # Preprocess input y
        y_processed = np.asarray(y_train).flatten()

        # Validate input data
        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("X_train contains NaN or Infinity values in its data.")
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("X_train contains NaN or Infinity values.")
        if not np.all(np.isfinite(y_processed)):
            raise ValueError("y_train contains NaN or Infinity values.")
        if num_samples != y_processed.shape[0]:
            raise ValueError(
                f"X_train samples ({num_samples}) must match y_train samples ({y_processed.shape[0]})."
            )

        # Identify unique classes
        self.classes = np.unique(y_processed)
        self.n_classes = len(self.classes)
        self.loss_history = []

        if self.n_classes < 2:
            raise ValueError("At least 2 unique class labels are required.")

        if self.n_classes == 2:
            # Binary classification
            # Initialize weights if None or mismatched shape
            if self.weights is None or self.weights.shape[0] != num_features:
                self.weights = np.zeros(num_features)
            self.b = 0.0

            for i in range(self.max_iter):
                # Compute gradients and update parameters
                grad_w, grad_b = self.grad(X_processed, y_processed)
                self.weights -= self.learning_rate * grad_w
                if self.fit_intercept:
                    self.b -= self.learning_rate * grad_b

                # Compute loss
                z = X_processed @ self.weights + (self.b if self.fit_intercept else 0)
                y_proba = self.sigmoid(z)
                loss = self.binary_ce(y_processed, y_proba)
                self.loss_history.append(loss)

                # Check numerical stability
                if not np.all(np.isfinite(self.weights)) or (self.fit_intercept and not np.isfinite(self.b)):
                    print(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training.")
                    break
                if not np.isfinite(loss):
                    print(f"Warning: Loss became NaN/Inf at epoch {i + 1}. Stopping training.")
                    break

                # Early stopping based on loss convergence
                if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    if self.verbose:
                        print(f"Early stopping at epoch {i+1}: Loss change ({abs(self.loss_history[-1] - self.loss_history[-2]):.6f}) below tolerance ({self.tol:.6f}).")
                    break

                # Print progress if verbose
                if self.verbose:
                    print(f"Epoch {i+1}/{self.max_iter} - Loss: {loss:.6f}, Avg Weight: {np.mean(self.weights):.6f}")

        else:
            # Multi-class classification using One-vs-Rest
            self.binary_classifiers = []
            if self.verbose:
                print(f"Training {self.n_classes} binary classifiers using One-vs-Rest (OvR) strategy...")

            # Aggregate loss for monitoring
            avg_loss_history = []

            for class_label in self.classes:
                if self.verbose:
                    print(f"  Training classifier for class {class_label} vs rest...")

                # Create binary labels: 1 for current class, 0 for others
                y_ovr = (y_processed == class_label).astype(int)

                # Train a binary classifier with same hyperparameters including regularization
                clf = BasicClassifier(
                    max_iter=self.max_iter,
                    learning_rate=self.learning_rate,
                    verbose=0,  # Suppress verbose for individual classifiers
                    fit_intercept=self.fit_intercept,
                    tol=self.tol,
                    penalty=self.penalty,
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio
                )
                clf.fit(X_processed, y_ovr)
                self.binary_classifiers.append(clf)

                # Aggregate loss from this classifier
                avg_loss_history.append(clf.loss_history)

            # Average loss across all OvR classifiers for monitoring
            if avg_loss_history:
                self.loss_history = [np.mean([lh[i] for lh in avg_loss_history if i < len(lh)]) 
                                   for i in range(max(len(lh) for lh in avg_loss_history))]

    def predict_proba(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class probabilities for test data.

        Args:
            X_test: Test features (n_samples, n_features), dense or sparse.

        Returns:
            np.ndarray: Predicted probabilities. For binary: (n_samples,). For multi-class: (n_samples, n_classes).

        Raises:
            ValueError: If model is not trained or invalid.
        """
        # Preprocess input
        if not issparse(X_test):
            X_processed = X_test.reshape(-1, 1) if X_test.ndim == 1 else np.asarray(X_test)
        else:
            X_processed = X_test

        if self.n_classes == 0:
            raise ValueError("Model not trained. Call fit() first.")

        if self.n_classes == 2:
            # Binary classification
            if self.weights is None:
                raise ValueError("Weights not initialized. Call fit() first.")
            z = X_processed @ self.weights + (self.b if self.fit_intercept else 0)
            return self.sigmoid(z)

        else:
            # Multi-class classification
            if not self.binary_classifiers:
                raise ValueError("OvR classifiers not trained. Call fit() first.")
            
            # Collect probabilities from each OvR classifier
            all_probas = np.zeros((X_processed.shape[0], self.n_classes))
            for i, clf in enumerate(self.binary_classifiers):
                all_probas[:, i] = clf.predict_proba(X_processed)
            return all_probas

    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels for test data.

        Args:
            X_test: Test features (n_samples, n_features), dense or sparse.

        Returns:
            np.ndarray: Predicted class labels (n_samples,).

        Raises:
            ValueError: If model is not trained or invalid.
        """
        probas = self.predict_proba(X_test)

        if self.n_classes == 2:
            # Binary classification: threshold at 0.5
            return (probas >= 0.5).astype(int)
        else:
            # Multi-class: select class with highest probability
            pred_indices = np.argmax(probas, axis=1)
            # Map indices to original class labels
            return np.array([self.classes[idx] for idx in pred_indices])
    
    def score(self, X_test: np.ndarray | spmatrix, y_test: np.ndarray) -> float:
        """
        Calculate accuracy score of the classifier.

        Args:
            X_test: Test features (n_samples, n_features), dense or sparse.
            y_test: True labels for test data (n_samples,).

        Returns:
            float: Accuracy score between 0 and 1.
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
