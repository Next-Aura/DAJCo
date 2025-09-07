import numpy as np
from typing import Literal, Optional
from scipy.sparse import issparse, spmatrix

try:
    from numba import jit as _numba_jit
    def _jit(*args, **kwargs):
        return _numba_jit(*args, **kwargs)
except Exception:
    def _jit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

class TBRegressor:
    def __init__(
            self,
            max_depth: Optional[int] = 5,
            min_samples_leaf: Optional[int] = 1,
            criterion: Literal['mse', 'friedman_mse', 'absolute_error', 'poisson'] = 'mse',
            max_features: Optional[int] = None,
            random_state: Optional[int] = None,
            min_samples_split: Optional[int] = 2,
            min_impurity_decrease: Optional[float] = 0.0
            ):
        """
        Initialize the TBRegressor.

        Args:
            max_depth (Optional[int]): The maximum depth of the tree. Defaults to 5.
            min_samples_leaf (Optional[int]): The minimum number of samples required to be at a leaf node. Defaults to 1.
            criterion (Literal): The function to measure the quality of a split. Supported criteria are
                'mse', 'friedman_mse', 'absolute_error', and 'poisson'. Defaults to 'mse'.
            max_features (Optional[int]): The number of features to consider when looking for the best split.
                If None, then all features are considered. Defaults to None.
            random_state (Optional[int]): Seed used by the random number generator. Defaults to None.
            min_samples_split (Optional[int]): The minimum number of samples required to split an internal node.
                Defaults to 2.
            min_impurity_decrease (Optional[float]): A node will be split if this split induces a decrease of the
                impurity greater than or equal to this value. Defaults to 0.0.

        Raises:
            ValueError: If criterion is not one of the valid options.
            ValueError: If min_samples_split is less than 2 * min_samples_leaf.
        """
        valid_criteria = ['mse', 'friedman_mse', 'absolute_error', 'poisson']
        if criterion not in valid_criteria:
            raise ValueError(f"Criterion must be one of {valid_criteria}")

        if max_depth is None or max_depth <= 0:
            max_depth = 5

        if min_samples_leaf is None or min_samples_leaf <= 0:
            min_samples_leaf = 1
        
        if min_samples_split is None or min_samples_split <= 0:
            min_samples_split = 2

        if min_impurity_decrease is None or min_impurity_decrease < 0:
            min_impurity_decrease = 0.0

        if 2 * min_samples_leaf < min_samples_split:
            raise ValueError("min_samples_split must be at least 2 * min_samples_leaf")

        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.tree = None
        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    @_jit(nopython=True)
    def variance(labels: np.ndarray) -> float:
        """
        Calculate the variance of the given labels.

        Args:
            labels (np.ndarray): Array of target values.

        Returns:
            float: Variance of the labels. Returns 0.0 if labels are empty.
        """
        if labels.size == 0:
            return 0.0
        mean = np.mean(labels)
        return np.mean((labels - mean) ** 2)

    @staticmethod
    def mean_absolute_error(labels: np.ndarray) -> float:
        """
        Calculate the mean absolute error of the given labels.

        Args:
            labels (np.ndarray): Array of target values.

        Returns:
            float: Mean absolute error of the labels. Returns 0.0 if labels are empty.
        """
        if labels.size == 0:
            return 0.0
        mean = np.mean(labels)
        return np.mean(np.abs(labels - mean))

    def _poisson_deviance(self, y: np.ndarray) -> float:
        """
        Calculate the Poisson deviance of the given labels.

        Args:
            y (np.ndarray): Array of target values.

        Returns:
            float: Poisson deviance of the labels. Returns 0.0 if labels are empty or mean is non-positive.

        Notes:
            Values in y should be non-negative for Poisson deviance to be valid.
        """
        if y.size == 0:
            return 0.0
        if np.any(y < 0):
            raise ValueError("Poisson deviance requires non-negative target values.")
        mean_y = np.mean(y)
        if mean_y <= 0:
            return 0.0
        return 2.0 * np.sum(y * np.log(np.maximum(y, 1e-9) / mean_y) - (y - mean_y))

    def _impurity(self, labels: np.ndarray) -> float:
        """
        Calculate the impurity of the given labels based on the selected criterion.

        Args:
            labels (np.ndarray): Array of target values.

        Returns:
            float: Impurity value according to the criterion.

        Raises:
            ValueError: If the criterion is unknown.
        """
        if self.criterion == 'mse':
            return self.variance(labels)
        elif self.criterion == 'friedman_mse':
            n = labels.size
            if n <= 1:
                return 0.0
            mse = self.variance(labels)
            return mse * (n / (n - 1))
        elif self.criterion == 'absolute_error':
            return self.mean_absolute_error(labels)
        elif self.criterion == 'poisson':
            return self._poisson_deviance(labels)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def criterion_score(self, y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """
        Calculate the impurity decrease (gain) from splitting the parent node into left and right children.

        Args:
            y (np.ndarray): Parent node target values.
            left_y (np.ndarray): Left child target values.
            right_y (np.ndarray): Right child target values.

        Returns:
            float: Impurity decrease (gain) from the split. Returns 0.0 if parent node is empty.
        """
        n = y.size
        if n == 0:
            return 0.0
        parent_imp = self._impurity(y)
        n_left = left_y.size
        n_right = right_y.size
        left_imp = self._impurity(left_y) if n_left > 0 else 0.0
        right_imp = self._impurity(right_y) if n_right > 0 else 0.0
        children_imp = (n_left * left_imp + n_right * right_imp) / n
        gain = parent_imp - children_imp
        return gain

    def find_best_split(self, X: np.ndarray | spmatrix, y: np.ndarray, feature_idx: int) -> tuple[float, float | None]:
        """
        Find the best split value for a given feature that maximizes impurity decrease.

        Args:
            X (np.ndarray or sparse matrix): Feature matrix.
            y (np.ndarray): Target values.
            feature_idx (int): Index of the feature to find the split for.

        Returns:
            tuple: (best_gain, best_value) where best_gain is the impurity decrease and best_value is the split threshold.
                Returns (None, None) if no valid split is found.
        """
        if issparse(X):
            feature_values = X[:, feature_idx].toarray().ravel()
        else:
            feature_values = X[:, feature_idx]

        unique_values = np.unique(feature_values)
        if len(unique_values) <= 1:
            return None, None
        
        if y.size < self.min_samples_split:
            return None, None

        if len(unique_values) > 100:
            num_points = 51 if issparse(X) else 101
            split_points = np.percentile(feature_values, np.linspace(0, 100, num_points))
            split_points = np.unique(split_points)
        else:
            split_points = (unique_values[:-1] + unique_values[1:]) / 2.0
        best_gain = -np.inf
        best_value = None
        n = y.size

        for value in split_points:
            left_mask = feature_values <= value
            right_mask = feature_values > value
            left_labels = y[left_mask]
            right_labels = y[right_mask]
            if left_labels.size < self.min_samples_leaf or right_labels.size < self.min_samples_leaf:
                continue
            gain = self.criterion_score(y, left_labels, right_labels)
            
            if gain > best_gain and gain >= self.min_impurity_decrease:
                best_gain = gain
                best_value = value

        if best_value is None or best_gain <= 0.0:
            return None, None
        return best_gain, best_value

    def find_best_feature_split(self, X: np.ndarray | spmatrix, y: np.ndarray) -> tuple[int | None, float | None, float]:
        """
        Find the best feature and split value that maximizes impurity decrease.

        Args:
            X (np.ndarray or sparse matrix): Feature matrix.
            y (np.ndarray): Target values.

        Returns:
            tuple: (best_feature, best_value, best_gain) where best_feature is the index of the best feature,
                best_value is the split threshold, and best_gain is the impurity decrease.
                Returns (None, None, None) if no valid split is found.
        """
        best_gain = -np.inf
        best_feature = None
        best_value = None
        n_features = X.shape[1]
        if self.max_features is not None and self.max_features < n_features:
            features = np.random.choice(n_features, self.max_features, replace=False)
        else:
            features = range(n_features)

        for feature_idx in features:
            result = self.find_best_split(X, y, feature_idx)
            if result is None:
                continue
            gain, value = result
            if gain is None:
                continue
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_value = value

        if best_feature is None:
            return None, None, None
        return best_feature, best_value, best_gain

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray, depth: int = 0) -> dict:
        """
        Build the decision tree regressor by recursively finding the best splits.

        Args:
            X (np.ndarray or sparse matrix): Feature matrix.
            y (np.ndarray): Target values.
            depth (int): Current depth of the tree. Defaults to 0.

        Returns:
            dict: A nested dictionary representing the tree structure with nodes containing
                'feature', 'threshold', 'left', 'right' or a leaf node with 'value'.
        """
        if not issparse(X):
            X = np.asarray(X)
        else:
            X = X.tocsr()
        y = np.asarray(y)

        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        if X.shape[0] != y.size:
            raise ValueError("X and y must have the same length")

        if depth >= self.max_depth or y.size <= self.min_samples_leaf or y.size < self.min_samples_split:
            return {"value": float(np.mean(y))}

        feature_idx, value, gain = self.find_best_feature_split(X, y)

        if feature_idx is None:
            return {"value": float(np.mean(y))}

        
        if gain < self.min_impurity_decrease:
            return {"value": float(np.mean(y))}

        col = X[:, feature_idx]
        if issparse(col):
            col = col.toarray().ravel()
        left_mask = col <= value
        right_mask = col > value

        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        if left_y.size == 0 or right_y.size == 0:
            return {"value": float(np.mean(y))}

        node = {
            "feature": int(feature_idx),
            "threshold": float(value),
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }

        if depth == 0:
            self.tree = node
        return node

    def _predict_single(self, x: np.ndarray | spmatrix, tree: dict=None) -> float:
        """
        Predict the target value for a single sample by traversing the tree.

        Args:
            x (np.ndarray or sparse matrix): Single sample feature vector.
            tree (dict, optional): The tree or subtree to use for prediction. Defaults to None (use full tree).

        Returns:
            float: Predicted target value.
        """
        if tree is None:
            tree = self.tree
        if issparse(x):
            x = x.toarray().ravel()
        else:
            x = np.asarray(x).ravel()

        if "value" in tree and "feature" not in tree:
            return tree["value"]

        feature_idx = tree["feature"]
        threshold = tree["threshold"]
        if x[feature_idx] <= threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict target values for multiple samples.

        Args:
            X (np.ndarray or sparse matrix): Feature matrix.

        Returns:
            np.ndarray: Array of predicted target values.
        """
        if not issparse(X):
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return np.array(predictions, dtype=float)

    def score(self, X: np.ndarray | spmatrix, y: np.ndarray) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.

        Args:
            X (np.ndarray or sparse matrix): Feature matrix.
            y (np.ndarray): True target values.

        Returns:
            float: R^2 score. Returns 0.0 if variance of y is zero.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v if v != 0 else 0.0
