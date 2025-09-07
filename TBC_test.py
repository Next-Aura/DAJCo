import time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from BasicModels.TBC import TBClassifier

def test_binary_classification(n_samples=1000, n_features=20, random_state=42):
    """
    Test TBC on a binary classification dataset.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features in the dataset.
        random_state: Seed for reproducibility.
    """
    # Generate synthetic binary classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2,
                              n_informative=15, n_redundant=5, random_state=random_state)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Test dense input
    print("Testing TBC on binary classification (dense input)...")
    clf_dense = TBClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_dense.fit(X_train, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Dense Input - Training time: {training_time:.4f} seconds")

    # Predict and evaluate
    y_pred_dense = clf_dense.predict(X_test)
    acc_dense = accuracy_score(y_test, y_pred_dense)
    print(f"Dense Input - Accuracy: {acc_dense:.4f}")

    # Test sparse input
    print("\nTesting TBC on binary classification (sparse input)...")
    X_train_sparse = csr_matrix(X_train)
    X_test_sparse = csr_matrix(X_test)
    clf_sparse = TBClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_sparse.fit(X_train_sparse, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Sparse Input - Training time: {training_time:.4f} seconds")

    # Predict and evaluate
    y_pred_sparse = clf_sparse.predict(X_test_sparse)
    acc_sparse = accuracy_score(y_test, y_pred_sparse)
    print(f"Sparse Input - Accuracy: {acc_sparse:.4f}")

    # Benchmark with scikit-learn's DecisionTreeClassifier
    print("\nBenchmarking with sklearn DecisionTreeClassifier...")
    clf_sklearn = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_sklearn.fit(X_train, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Sklearn DecisionTreeClassifier - Training time: {training_time:.4f} seconds")
    y_pred_sklearn = clf_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn DecisionTreeClassifier - Accuracy: {acc_sklearn:.4f}")

def test_multiclass_classification(n_samples=1000, n_features=20, n_classes=4, random_state=42):
    """
    Test TBC on a multi-class classification dataset.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features in the dataset.
        n_classes: Number of classes in the dataset.
        random_state: Seed for reproducibility.
    """
    # Generate synthetic multi-class classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                              n_informative=15, n_redundant=5, n_clusters_per_class=1,
                              random_state=random_state)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Scale features for better convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test dense input
    print("\nTesting TBC on multi-class classification (dense input)...")
    clf_dense = TBClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_dense.fit(X_train, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Dense Input - Training time: {training_time:.4f} seconds")

    # Predict and evaluate
    y_pred_dense = clf_dense.predict(X_test)
    acc_dense = accuracy_score(y_test, y_pred_dense)
    print(f"Dense Input - Accuracy: {acc_dense:.4f}")

    # Test sparse input
    print("\nTesting TBC on multi-class classification (sparse input)...")
    X_train_sparse = csr_matrix(X_train)
    X_test_sparse = csr_matrix(X_test)
    clf_sparse = TBClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_sparse.fit(X_train_sparse, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Sparse Input - Training time: {training_time:.4f} seconds")

    # Predict and evaluate
    y_pred_sparse = clf_sparse.predict(X_test_sparse)
    acc_sparse = accuracy_score(y_test, y_pred_sparse)
    print(f"Sparse Input - Accuracy: {acc_sparse:.4f}")

    # Benchmark with scikit-learn's DecisionTreeClassifier
    print("\nBenchmarking with sklearn DecisionTreeClassifier...")
    clf_sklearn = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)
    start = time.perf_counter()
    clf_sklearn.fit(X_train, y_train)
    end = time.perf_counter()
    training_time = end - start
    print(f"Sklearn DecisionTreeClassifier - Training time: {training_time:.4f} seconds")
    y_pred_sklearn = clf_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn DecisionTreeClassifier - Accuracy: {acc_sklearn:.4f}")

if __name__ == "__main__":
    """
    Main execution block to run TBC performance tests.
    Tests both binary and multi-class classification with dense and sparse inputs.
    Compares results with scikit-learn's DecisionTreeClassifier for benchmarking.
    """
    print("=== TBC Performance Test ===")
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run binary classification test
    print("\n--- Binary Classification Test ---")
    test_binary_classification(n_samples=1000, n_features=20, random_state=42)

    # Run multi-class classification test
    print("\n--- Multi-Class Classification Test ---")
    test_multiclass_classification(n_samples=1000, n_features=20, n_classes=4, random_state=42)