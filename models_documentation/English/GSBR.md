# GSBR – Gradient Supported Basic Regressor

## Overview
GSBR (Gradient Supported Basic Regressor) is a custom linear regression model that supports various regularization techniques including L1, L2, and ElasticNet. It uses mean squared error (MSE) or root mean squared error (RMSE) as loss functions, optimized using gradient descent.

## Installation & Requirements
```bash
pip install numpy scipy
```

## Mathematical Formulation

### Linear Regression
The model uses linear regression with the prediction function:

**Prediction**: 
$$\hat{y} = Xw + b$$

**Mean Squared Error (MSE)**:
$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**:
$$L_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

### Regularization
- **L1 (Lasso)**: $\alpha \sum |w_i|$
- **L2 (Ridge)**: $\alpha \sum w_i^2$
- **Elastic Net**: $\alpha \left[ l1\_ratio \cdot \sum |w_i| + (1 - l1\_ratio) \cdot \sum w_i^2 \right]$

### Gradients
**Weight Gradient**:
$$\frac{\partial L}{\partial w} = \frac{2}{N} X^T (Xw + b - y) + \text{regularization gradient}$$

**Bias Gradient**:
$$\frac{\partial L}{\partial b} = \frac{2}{N} \sum (Xw + b - y)$$

## Key Features
- ✅ Supports L1, L2, and ElasticNet regularization
- ✅ Works with dense and sparse matrix inputs
- ✅ Loss functions: MSE or RMSE
- ✅ Includes early stopping and verbose training output
- ✅ Comprehensive input validation
- ✅ Gradient descent optimization with learning rate control

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `100` | Maximum number of gradient descent iterations |
| `learning_rate` | `float` | `0.01` | Step size for gradient descent updates |
| `verbose` | `int` | `0` | If 1, print training progress (epoch, weights, bias, loss) |
| `penalty` | `Literal['l1', 'l2', 'elasticnet'] \| None` | `'l2'` | Type of regularization |
| `alpha` | `float` | `0.0001` | Regularization strength |
| `l1_ratio` | `float` | `0.5` | Mixing parameter for ElasticNet (0 = L2, 1 = L1) |
| `fit_intercept` | `bool` | `True` | Whether to include a bias term (intercept) |
| `tol` | `float` | `0.0001` | Tolerance for early stopping based on loss convergence |
| `loss` | `Literal['mse', 'rmse']` | `'mse'` | Loss function to minimize |

## Model Attributes (After Fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned feature weights |
| `b` | `float` | Bias term (intercept) |
| `loss_history` | `List[float]` | Loss values at each iteration |

## API Reference

### `BasicRegressor.__init__()`
Initialize the regressor with hyperparameters.

### `BasicRegressor.fit(X_train, y_train)`
Train the regressor on the given data.

**Parameters**:
- `X_train`: Training features (n_samples, n_features), dense or sparse
- `y_train`: Training target values (n_samples,)

**Raises**:
- `ValueError`: If input data contains NaN/Inf or mismatched shapes
- `TypeError`: If input data contains non-numeric types

### `BasicRegressor.predict(X_test)`
Predict target values for test data.

**Returns**: (n_samples,) predicted values

**Raises**:
- `ValueError`: If model is not trained

### `BasicRegressor.loss_score()`
Get the mean loss over all iterations.

**Returns**: Mean loss value

**Raises**:
- `ValueError`: If no training history available

## Usage Examples

### Basic Regression with L2 Regularization
```python
from BasicModels.GSBR import BasicRegressor
import numpy as np
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Initialize and train model
model = BasicRegressor(
    max_iter=1000,
    learning_rate=0.01,
    verbose=1,
    penalty='l2',
    alpha=0.1,
    fit_intercept=True,
    tol=1e-4,
    loss='mse'
)
model.fit(X, y)

# Make predictions
preds = model.predict(X)
print(f"Final weights: {model.weights}")
print(f"Final bias: {model.b}")
print(f"Mean loss: {model.loss_score():.6f}")
```

### ElasticNet Regularization with RMSE Loss
```python
from BasicModels.GSBR import BasicRegressor
from sklearn.preprocessing import StandardScaler

# Scale features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = BasicRegressor(
    max_iter=2000,
    learning_rate=0.005,
    penalty='elasticnet',
    alpha=0.05,
    l1_ratio=0.3,
    loss='rmse',
    verbose=1
)
model.fit(X_scaled, y)

print(f"ElasticNet model trained with RMSE: {model.loss_score():.4f}")
```

### Sparse Matrix Support
```python
from BasicModels.GSBR import BasicRegressor
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse data
X_sparse = csr_matrix(np.random.randn(100, 10))
y = np.random.randn(100)

model = BasicRegressor(
    max_iter=500,
    learning_rate=0.02,
    verbose=0
)
model.fit(X_sparse, y)

print("Sparse matrix training completed successfully!")
```

### No Regularization (Ordinary Least Squares)
```python
model = BasicRegressor(
    max_iter=1000,
    learning_rate=0.01,
    penalty=None,
    fit_intercept=True,
    verbose=1
)
model.fit(X, y)

print(f"OLS model - Final loss: {model.loss_history[-1]:.6f}")
```

## Best Practices

### Hyperparameter Tuning
- Start with `learning_rate=0.01-0.1` and adjust based on convergence
- Use `max_iter=1000-5000` for complex problems
- Set `tol=1e-4` for reasonable early stopping
- For regularization: start with small `alpha` values (0.001-0.1)
- For ElasticNet: `l1_ratio=0.5` provides balanced L1/L2 mix

### Data Preprocessing
- Scale features to mean=0, std=1 for better convergence
- Handle missing values before training
- Consider feature engineering for better performance

### Monitoring Training
```python
import matplotlib.pyplot as plt

model = BasicRegressor(max_iter=1000, learning_rate=0.01, verbose=0)
model.fit(X_train, y_train)

plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.grid(True)
plt.show()
```

## Error Handling

The model includes comprehensive error checking:
- Input validation for NaN/Inf values
- Type checking for numeric data
- Shape consistency checks
- Parameter validation (positive values, valid ranges)
- Early stopping for numerical instability

## Performance Considerations

- **Memory**: Efficient with both dense and sparse matrices
- **Speed**: Batch gradient descent - suitable for medium-sized datasets
- **Scalability**: For very large datasets, consider mini-batch variants

## Comparison with scikit-learn

| Feature | GSBR | scikit-learn LinearRegression/ElasticNet |
|---------|------|------------------------------------------|
| Custom regularization | ✅ | ✅ |
| Sparse matrix support | ✅ | ✅ |
| Gradient descent | ✅ | ❌ (uses analytical solutions) |
| Early stopping | ✅ | ✅ |
| Learning rate control | ✅ | ❌ |
| Multiple solvers | ❌ | ✅ |
| Cross-validation | ❌ | ✅ |

## License

This implementation is provided as part of the BasicModels package.
