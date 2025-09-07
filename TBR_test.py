import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from BasicModels.TBR import TBRegressor

def test_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42):
    """
    Test TBR on a regression dataset with different criteria and input formats.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features in the dataset.
        noise: Standard deviation of Gaussian noise added to the output.
        random_state: Seed for reproducibility.
    """
    # Generate synthetic regression dataset
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise,
                           n_informative=15, random_state=random_state)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Scale features for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define criteria to test
    criteria = ['mse', 'friedman_mse', 'absolute_error', 'poisson']

    for criterion in criteria:
        # For Poisson, ensure non-negative targets
        if criterion == 'poisson':
            y_train_criterion = y_train - np.min(y_train) + 1  # Shift to positive
            y_test_criterion = y_test - np.min(y_test) + 1
        else:
            y_train_criterion = y_train
            y_test_criterion = y_test

        print(f"\nTesting TBR with criterion={criterion} (dense input)...")
        # Configure TBR with specific criterion
        reg_dense = TBRegressor(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion=criterion,
            random_state=random_state
        )

        # Time the training
        start_time = time.time()
        reg_dense.fit(X_train, y_train_criterion)
        train_time = time.time() - start_time

        # Time the prediction
        start_time = time.time()
        y_pred_dense = reg_dense.predict(X_test)
        pred_time = time.time() - start_time

        # Evaluate
        mse_dense = mean_squared_error(y_test_criterion, y_pred_dense)
        rmse_dense = np.sqrt(mse_dense)
        mae_dense = mean_absolute_error(y_test_criterion, y_pred_dense)
        r2_dense = r2_score(y_test_criterion, y_pred_dense)
        print(f"Dense Input - MSE: {mse_dense:.4f}, RMSE: {rmse_dense:.4f}, MAE: {mae_dense:.4f}, R²: {r2_dense:.4f}")
        print(f"Training time: {train_time:.4f}s, Prediction time: {pred_time:.4f}s")

        # Test sparse input
        print(f"\nTesting TBR with criterion={criterion} (sparse input)...")
        X_train_sparse = csr_matrix(X_train)
        X_test_sparse = csr_matrix(X_test)
        reg_sparse = TBRegressor(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion=criterion,
            random_state=random_state
        )

        # Time the training
        start_time = time.time()
        reg_sparse.fit(X_train_sparse, y_train_criterion)
        train_time = time.time() - start_time

        # Time the prediction
        start_time = time.time()
        y_pred_sparse = reg_sparse.predict(X_test_sparse)
        pred_time = time.time() - start_time

        # Evaluate
        mse_sparse = mean_squared_error(y_test_criterion, y_pred_sparse)
        rmse_sparse = np.sqrt(mse_sparse)
        mae_sparse = mean_absolute_error(y_test_criterion, y_pred_sparse)
        r2_sparse = r2_score(y_test_criterion, y_pred_sparse)
        print(f"Sparse Input - MSE: {mse_sparse:.4f}, RMSE: {rmse_sparse:.4f}, MAE: {mae_sparse:.4f}, R²: {r2_sparse:.4f}")
        print(f"Training time: {train_time:.4f}s, Prediction time: {pred_time:.4f}s")

    # Benchmark with scikit-learn's DecisionTreeRegressor
    print("\nBenchmarking with sklearn DecisionTreeRegressor...")
    reg_sklearn = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=random_state)

    # Time the training
    start_time = time.time()
    reg_sklearn.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Time the prediction
    start_time = time.time()
    y_pred_sklearn = reg_sklearn.predict(X_test)
    pred_time = time.time() - start_time

    # Evaluate
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    rmse_sklearn = np.sqrt(mse_sklearn)
    mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    print(f"Sklearn DecisionTreeRegressor - MSE: {mse_sklearn:.4f}, RMSE: {rmse_sklearn:.4f}, MAE: {mae_sklearn:.4f}, R²: {r2_sklearn:.4f}")
    print(f"Training time: {train_time:.4f}s, Prediction time: {pred_time:.4f}s")

if __name__ == "__main__":
    """
    Main execution block to run TBR performance tests.
    Tests regression with different criteria (mse, friedman_mse, absolute_error, poisson) and input formats (dense, sparse).
    Compares results with scikit-learn's DecisionTreeRegressor for benchmarking.
    """
    print("=== TBR Performance Test ===")
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run regression test
    print("\n--- Regression Test ---")
    test_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42)
