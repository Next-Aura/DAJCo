import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from BasicModels.GSBR import BasicRegressor  # Assumes GSBR.py is in the same directory

def test_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42):
    """
    Test GSBR on a regression dataset with different regularization types and input formats.

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
    
    # Scale features for better convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define regularization types to test
    penalties = [None, "l1", "l2", "elasticnet"]
    
    for penalty in penalties:
        print(f"\nTesting GSBR with penalty={penalty} (dense input)...")
        # Configure GSBR with specific penalty
        clf_dense = BasicRegressor(
            max_iter=1000,
            learning_rate=0.01,
            verbose=1,
            penalty=penalty,
            alpha=0.1 if penalty else 0.0,  # Use small regularization strength
            l1_ratio=0.5 if penalty == "elasticnet" else 0.0,
            fit_intercept=True,
            tol=1e-4,
            loss="mse"
        )
        clf_dense.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred_dense = clf_dense.predict(X_test)
        mse_dense = mean_squared_error(y_test, y_pred_dense)
        rmse_dense = np.sqrt(mse_dense)
        r2_dense = r2_score(y_test, y_pred_dense)
        print(f"Dense Input - MSE: {mse_dense:.4f}, RMSE: {rmse_dense:.4f}, R²: {r2_dense:.4f}")
        
        # Test sparse input
        print(f"\nTesting GSBR with penalty={penalty} (sparse input)...")
        X_train_sparse = csr_matrix(X_train)
        X_test_sparse = csr_matrix(X_test)
        clf_sparse = BasicRegressor(
            max_iter=1000,
            learning_rate=0.01,
            verbose=1,
            penalty=penalty,
            alpha=0.1 if penalty else 0.0,
            l1_ratio=0.5 if penalty == "elasticnet" else 0.0,
            fit_intercept=True,
            tol=1e-4,
            loss="mse"
        )
        clf_sparse.fit(X_train_sparse, y_train)
        
        # Predict and evaluate
        y_pred_sparse = clf_sparse.predict(X_test_sparse)
        mse_sparse = mean_squared_error(y_test, y_pred_sparse)
        rmse_sparse = np.sqrt(mse_sparse)
        r2_sparse = r2_score(y_test, y_pred_sparse)
        print(f"Sparse Input - MSE: {mse_sparse:.4f}, RMSE: {rmse_sparse:.4f}, R²: {r2_sparse:.4f}")
    
    # Benchmark with scikit-learn's LinearRegression (no regularization)
    print("\nBenchmarking with sklearn LinearRegression...")
    clf_lr = LinearRegression(fit_intercept=True)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"Sklearn LinearRegression - MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}")
    
    # Benchmark with scikit-learn's ElasticNet (for comparison with regularized GSBR)
    print("\nBenchmarking with sklearn ElasticNet...")
    clf_en = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, random_state=random_state)
    clf_en.fit(X_train, y_train)
    y_pred_en = clf_en.predict(X_test)
    mse_en = mean_squared_error(y_test, y_pred_en)
    rmse_en = np.sqrt(mse_en)
    r2_en = r2_score(y_test, y_pred_en)
    print(f"Sklearn ElasticNet - MSE: {mse_en:.4f}, RMSE: {rmse_en:.4f}, R²: {r2_en:.4f}")

if __name__ == "__main__":
    """
    Main execution block to run GSBR performance tests.
    Tests regression with different regularization types (None, L1, L2, ElasticNet) and input formats (dense, sparse).
    Compares results with scikit-learn's LinearRegression and ElasticNet for benchmarking.
    """
    print("=== GSBR Performance Test ===")
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run regression test
    print("\n--- Regression Test ---")
    test_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42)