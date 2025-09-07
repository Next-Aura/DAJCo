import numpy as np
from typing import Literal
from scipy.sparse import issparse, spmatrix
import warnings

try:
    from numba import jit
except ImportError:
    jit = lambda f: f

class TBClassifier():
    """
    A Tree-Based Classifier implementing a decision tree algorithm for classification tasks.

    This classifier builds a binary decision tree by recursively splitting the dataset based on
    feature values to minimize impurity. It supports Gini impurity, entropy, and log loss as
    splitting criteria. The tree can handle both dense and sparse input data.

    Parameters are validated and defaulted if invalid values are provided. The classifier
    includes pruning based on minimum samples per leaf, minimum samples per split, and
    minimum impurity decrease to prevent overfitting.

    Attributes:
        max_depth (int): Maximum depth of the decision tree.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        criterion (str): The impurity criterion ('gini', 'entropy', or 'log_loss').
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_impurity_decrease (float): Minimum impurity decrease required for a split.
        tree (dict): The trained decision tree structure.
    """
    def __init__(
            self,
            max_depth: int | None=2,
            min_samples_leaf: int | None=5,
            criterion: Literal['gini', 'entropy', 'log_loss'] | None='gini',
            min_samples_split: int | None=2,
            min_impurity_decrease: float | None=0.0,
            random_state: int | None=None
            ):
        """
        Initialize the TBClassifier with specified parameters.

        Parameters:
            max_depth (int, optional): Maximum depth of the tree. If None or <=0, defaults to 2.
            min_samples_leaf (int, optional): Minimum number of samples required to be at a leaf node. If None or <=0, defaults to 5.
            criterion (str, optional): The function to measure the quality of a split. Options are 'gini', 'entropy', 'log_loss'. Defaults to 'gini'.
            min_samples_split (int, optional): Minimum number of samples required to split an internal node. If None or <=0, defaults to 2.
            min_impurity_decrease (float, optional): Minimum impurity decrease required for a split. If None or <0, defaults to 0.0.
            random_state (int, optional): Seed for reproducibility. Not currently used.

        Raises:
            ValueError: If criterion is not one of the allowed values or if min_samples_split < 2 * min_samples_leaf.
            ValueError: If criterion is invalid.
            Warning: If any parameter is invalid, a warning is issued and a default value is set.
        """

        if criterion not in ('gini', 'entropy', 'log_loss'):
            raise ValueError("Criterion must be one of 'gini', 'entropy', or 'log_loss'")

        if max_depth is None or max_depth <= 0:
            warnings.warn("max_depth should be a positive integer. Setting to default value of 2.")
            max_depth = 2

        if min_samples_leaf is None or min_samples_leaf <= 0:
            warnings.warn("min_samples_leaf should be a positive integer. Setting to default value of 5.")
            min_samples_leaf = 5

        if min_samples_split is None or min_samples_split <= 0:
            warnings.warn("min_samples_split should be a positive integer. Setting to default value of 2.")
            min_samples_split = 2

        if min_impurity_decrease is None or min_impurity_decrease < 0:
            warnings.warn("min_impurity_decrease should be a non-negative float. Setting to default value of 0.0.")
            min_impurity_decrease = 0.0

        if 2 * min_samples_leaf < min_samples_split:
            raise ValueError("min_samples_split must be at least 2 * min_samples_leaf")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

    @staticmethod
    @jit(nopython=True)
    def gini_impurity(labels: np.ndarray) -> float:
        """
        Calculate the Gini impurity for a set of labels.

        Gini impurity measures the impurity of a node in a decision tree.
        It is defined as 1 - sum(p_i^2) where p_i is the proportion of samples
        of class i in the node.

        Parameters:
            labels (array-like): Array of class labels.

        Returns:
            float: The Gini impurity value.
        """
        if len(labels) == 0:
            return 0.0

        labels = labels.astype(np.int32)
        max_label = labels.max() if len(labels) > 0 else 0
        counts = np.bincount(labels, minlength=max_label + 1)
        probs = counts / len(labels)
        gini = 1.0 - np.sum(probs ** 2)
        return gini

    @staticmethod
    @jit(nopython=True)
    def entropy(labels: np.ndarray) -> float:
        """
        Calculate the entropy for a set of labels.

        Entropy measures the impurity of a node in a decision tree.
        It is defined as -sum(p_i * log2(p_i)) where p_i is the proportion of samples
        of class i in the node.

        Parameters:
            labels (array-like): Array of class labels.

        Returns:
            float: The entropy value.
        """
        if len(labels) == 0:
            return 0.0

        labels = labels.astype(np.int32)
        max_label = labels.max() if len(labels) > 0 else 0
        counts = np.bincount(labels, minlength=max_label + 1)
        probs = counts / len(labels)

        entropy_val = 0.0
        for p in probs:
            if p > 0:
                entropy_val -= p * np.log2(p)
        return entropy_val

    @staticmethod
    @jit(nopython=True)
    def log_loss(labels: np.ndarray) -> float:
        """
        Calculate the log loss (cross-entropy) for a set of labels.

        Log loss measures the impurity of a node in a decision tree.
        It is defined as -sum(p_i * log(p_i)) where p_i is the proportion of samples
        of class i in the node.

        Parameters:
            labels (array-like): Array of class labels.

        Returns:
            float: The log loss value.
        """
        if len(labels) == 0:
            return 0.0

        labels = labels.astype(np.int32)
        max_label = labels.max() if len(labels) > 0 else 0
        counts = np.bincount(labels, minlength=max_label + 1)
        probs = counts / len(labels)

        log_loss_val = 0.0
        for p in probs:
            if p > 0:
                log_loss_val -= p * np.log(p)
        return log_loss_val

    def compute_variance_sparse(self, X: spmatrix) -> np.ndarray:
        """
        Compute the variance of each feature column in a sparse matrix.

        This method calculates the variance for each column of a sparse matrix X.
        Variance is computed as E[X^2] - (E[X])^2.

        Parameters:
            X (scipy.sparse matrix): Sparse input feature matrix.

        Returns:
            np.ndarray: Array of variances for each feature column.
        """
        variances = []
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mean = col_data.mean()
            mean_sq = (col_data.multiply(col_data)).mean()
            var = mean_sq - mean**2
            variances.append(var)
        return np.array(variances)

    def find_best_split(self, X: np.ndarray | spmatrix, y: np.ndarray, feature_idx: int) -> tuple[float, float | None]:
        """
        Find the best split value for a given feature to minimize impurity.

        This method evaluates potential split points for the specified feature index
        and returns the split value that results in the lowest weighted impurity.

        Parameters:
            X (array-like or sparse matrix): Feature matrix.
            y (array-like): Target labels.
            feature_idx (int): Index of the feature to split on.

        Returns:
            tuple: (best_impurity, best_value)
                best_impurity (float): The lowest weighted impurity found.
                best_value (float or None): The feature value to split on, or None if no valid split.
        """
        if len(y) < self.min_samples_split:
            return float('inf'), None

        if issparse(X):
            feature_values = X[:, feature_idx].toarray().ravel()
        else:
            feature_values = X[:, feature_idx]

        unique_values = np.unique(feature_values)
        if len(unique_values) <= 1:
            return float('inf'), None

        if len(unique_values) > 100:
            num_points = 51 if issparse(X) else 101
            split_points = np.percentile(feature_values, np.linspace(0, 100, num_points))
            split_points = np.unique(split_points)
        else:
            split_points = (unique_values[:-1] + unique_values[1:]) / 2.0

        best_impurity = float('inf')
        best_value = None

        for value in split_points:
            left_mask = feature_values <= value
            right_mask = feature_values > value

            left_labels = y[left_mask]
            right_labels = y[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            if self.criterion == 'gini':
                left_impurity = self.gini_impurity(left_labels)
                right_impurity = self.gini_impurity(right_labels)
            elif self.criterion == 'entropy':
                left_impurity = self.entropy(left_labels)
                right_impurity = self.entropy(right_labels)
            elif self.criterion == 'log_loss':
                left_impurity = self.log_loss(left_labels)
                right_impurity = self.log_loss(right_labels)

            weighted_impurity = (len(left_labels) / len(y)) * left_impurity + \
                                (len(right_labels) / len(y)) * right_impurity

            if self.criterion == 'gini':
                current_impurity = self.gini_impurity(y)
            elif self.criterion == 'entropy':
                current_impurity = self.entropy(y)
            elif self.criterion == 'log_loss':
                current_impurity = self.log_loss(y)

            if weighted_impurity < best_impurity and (current_impurity - weighted_impurity) >= self.min_impurity_decrease:
                best_impurity = weighted_impurity
                best_value = value

        return best_impurity, best_value
    
    def find_best_feature_split(self, X: np.ndarray | spmatrix, y: np.ndarray) -> tuple[int | None, float | None, float]:
        """
        Find the best feature and split value to minimize impurity.

        This method iterates over all features and finds the best split point
        that results in the lowest impurity.

        Parameters:
            X (array-like or sparse matrix): Feature matrix.
            y (array-like): Target labels.

        Returns:
            tuple: (best_feature, best_value, best_impurity)
                best_feature (int or None): Index of the best feature to split on, or None if no valid split.
                best_value (float or None): The feature value to split on.
                best_impurity (float): The impurity value of the best split.
        """
        best_impurity = float('inf')
        best_feature = None
        best_value = None

        if issparse(X):
            variance = self.compute_variance_sparse(X)
        else:
            variance = np.var(X, axis=0)

        if np.all(variance == 0):
            return None, None, float('inf')

        for feature_idx in range(X.shape[1]):
            if variance[feature_idx] == 0:
                continue

            impurity, value = self.find_best_split(X, y, feature_idx)

            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = feature_idx
                best_value = value

        return best_feature, best_value, best_impurity

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray, depth=0) -> dict:
        """
        Fit the decision tree to the training data recursively.

        This method builds the decision tree by recursively finding the best splits
        until stopping criteria are met (max depth, minimum samples, or pure node).

        Parameters:
            X (array-like or sparse matrix): Feature matrix.
            y (array-like): Target labels.
            depth (int, optional): Current depth of the tree during recursion. Defaults to 0.

        Returns:
            dict: A nested dictionary representing the decision tree.
                  Leaf nodes contain a "label" key with the predicted class.
        """
        if not issparse(X):
            X = np.asarray(X)
        else:
            X = X.tocsr()

        y = np.asarray(y)
        
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("X and y cannot be empty")
                
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
        
        except TypeError:
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("X and y cannot be empty")

            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same length")
            
        if len(np.unique(y)) == 1:
            return {"label": int(y[0])}

        if depth >= self.max_depth or len(y) < self.min_samples_leaf or len(y) < self.min_samples_split:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        feature_idx, value, impurity_value = self.find_best_feature_split(X, y)

        if feature_idx is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        col = X[:, feature_idx]
        if issparse(col):
            col = col.toarray().ravel()
        left_mask = col <= value
        right_mask = col > value
            
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        if len(left_y) == 0 or len(right_y) == 0:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        self.tree = {
            "feature": feature_idx,
            "value": value,
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }
        
        return self.tree
    
    def _predict_single(self, x: np.ndarray, tree: dict=None) -> int:
        """
        Predict the class label for a single sample using the decision tree.

        Parameters:
            x (array-like or sparse matrix): Single sample feature vector.
            tree (dict, optional): The decision tree or subtree to use for prediction. Defaults to the full tree.

        Returns:
            int: Predicted class label.
        """
        if tree is None:
            tree = self.tree

        if issparse(x):
            x = x.toarray().ravel()

        else:
            x = np.asarray(x).ravel()

        if "label" in tree:
            return tree["label"]

        feature_idx = tree["feature"]
        value = tree["value"]

        if x[feature_idx] <= value:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels for multiple samples.

        Parameters:
            X (array-like or sparse matrix): Feature matrix.

        Returns:
            np.ndarray: Array of predicted class labels.
        """
        if not issparse(X):
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return np.array(predictions)
