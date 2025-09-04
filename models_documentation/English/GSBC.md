# GSBC – Gradient Supported Basic Classifier

## Overview
GSBC (Gradient Supported Basic Classifier) is a custom-built classification model using logistic regression with gradient descent optimization. It supports both binary and multi-class classification using the One-vs-Rest (OvR) strategy.

## Installation & Requirements
```bash
pip install numpy scipy
```

## Mathematical Formulation

### Binary Classification
The model uses logistic regression with the sigmoid function:

**Sigmoid Function**: 
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Binary Cross-Entropy Loss**:
$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

**Gradients**:
$$\frac{\partial L}{\partial w} = \frac{1}{N} X^T (p - y)$$
$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum (p - y)$$

### Regularization
- **L1 (Lasso)**: $\alpha \sum |w_i|$
- **L2 (Ridge)**: $\alpha \sum w_i^2$
- **Elastic Net**: $\alpha \left[ l1\_ratio \cdot \sum |w_i| + (1 - l1\_ratio) \cdot \sum w_i^2 \right]$

## Key Features
- ✅ Supports binary and multi-class classification
- ✅ Handles dense and sparse matrices
- ✅ Uses Binary Cross Entropy Loss
- ✅ Optimized with Batch Gradient Descent
- ✅ Includes logging and early stopping features
- ✅ Supports L1, L2, and Elastic Net regularization
- ✅ Comprehensive input validation
- ✅ One-vs-Rest strategy for multi-class problems

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `1000` | Maximum number of gradient descent iterations |
| `learning_rate` | `float` | `0.001` | Step size for gradient descent updates |
| `verbose` | `int` | `0` | If 1, print training progress (epoch, loss, etc.) |
| `fit_intercept` | `bool` | `True` | Whether to include a bias term (intercept) |
| `tol` | `float` | `0.0001` | Tolerance for early stopping based on loss convergence |
| `penalty` | `Literal['l1', 'l2', 'elasticnet'] \| None` | `None` | Type of regularization |
| `alpha` | `float` | `0.001` | Regularization strength (used if penalty is not None) |
| `l1_ratio` | `float` | `0.5` | Mixing parameter for elastic net (0 <= l1_ratio <= 1) |

## Model Attributes (After Fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned feature weights |
| `b` | `float` | Bias term (intercept) |
| `loss_history` | `List[float]` | Loss values at each iteration |
| `classes` | `np.ndarray` | Unique class labels found during fitting |
| `n_classes` | `int` | Number of unique classes |
| `binary_classifiers` | `List[BasicClassifier]` | OvR classifiers for multi-class problems |

## API Reference

### `BasicClassifier.__init__()`
Initialize the classifier with hyperparameters.

### `BasicClassifier.fit(X_train, y_train)`
Train the classifier on the given data.

**Parameters**:
- `X_train`: Training features (n_samples, n_features), dense or sparse
- `y_train`: Training target labels (n_samples,)

**Raises**:
- `ValueError`: If input data contains NaN/Inf or mismatched shapes

### `BasicClassifier.predict_proba(X_test)`
Predict class probabilities for test data.

**Returns**:
- For binary: (n_samples,) probabilities
- For multi-class: (n_samples, n_classes) probability matrix

### `BasicClassifier.predict(X_test)`
Predict class labels for test data.

**Returns**: (n_samples,) predicted class labels

### `BasicClassifier.score(X_test, y_test)`
Calculate accuracy score of the classifier.

**Returns**: Accuracy score between 0 and 1

## Usage Examples

### Binary Classification
```python
from BasicModels.GSBC import BasicClassifier
import numpy as np

# Create sample data
X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Initialize and train model
model = BasicClassifier(
    max_iter=500, 
    learning_rate=0.01, 
    verbose=1,
    penalty='l2',
    alpha=0.1
)
model.fit(X, y)

# Make predictions
preds = model.predict(X)
probas = model.predict_proba(X)
accuracy = model.score(X, y)

print(f"Accuracy: {accuracy:.4f}")
print(f"Final loss: {model.loss_history[-1]:.6f}")
```

### Multi-class Classification
```python
from BasicModels.GSBC import BasicClassifier
from sklearn.datasets import make_classification
import numpy as np

# Create multi-class data
X, y = make_classification(
    n_samples=200, 
    n_features=4, 
    n_classes=3, 
    n_informative=3,
    random_state=42
)

# Train with regularization
model = BasicClassifier(
    max_iter=1000,
    learning_rate=0.005,
    verbose=1,
    penalty='elasticnet',
    alpha=0.01,
    l1_ratio=0.3
)
model.fit(X, y)

# Predict and evaluate
preds = model.predict(X)
accuracy = np.mean(preds == y)
print(f"Multi-class accuracy: {accuracy:.4f}")
print(f"Classes: {model.classes}")
```

### With Sparse Matrices
```python
from BasicModels.GSBC import BasicClassifier
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse data
X_sparse = csr_matrix(np.random.randn(100, 10))
y = (np.random.rand(100) > 0.5).astype(int)

model = BasicClassifier(max_iter=300, learning_rate=0.02)
model.fit(X_sparse, y)

print("Sparse matrix training completed!")
```

## Best Practices

### Hyperparameter Tuning
- Start with `learning_rate=0.01` and adjust based on convergence
- Use `max_iter=1000-5000` for complex problems
- Set `tol=1e-4` for reasonable early stopping
- For regularization: start with small `alpha` values (0.001-0.1)

### Data Preprocessing
- Scale features to mean=0, std=1 for better convergence
- Handle missing values before training
- For multi-class problems, ensure balanced classes or use class weights

### Monitoring Training
```python
# Monitor training progress
import matplotlib.pyplot as plt

model = BasicClassifier(max_iter=1000, learning_rate=0.01, verbose=1)
model.fit(X_train, y_train)

plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.show()
```

## Error Handling

The model includes comprehensive error checking:
- Input validation for NaN/Inf values
- Shape consistency checks
- Parameter validation (positive values, valid ranges)
- Early stopping for numerical instability

## Performance Considerations

- **Memory**: Efficient with both dense and sparse matrices
- **Speed**: Batch gradient descent - suitable for medium-sized datasets
- **Scalability**: For very large datasets, consider mini-batch variants

## Comparison with scikit-learn

| Feature | GSBC | scikit-learn LogisticRegression |
|---------|------|--------------------------------|
| Custom regularization | ✅ | ✅ |
| Sparse matrix support | ✅ | ✅ |
| Multi-class OvR | ✅ | ✅ |
| Early stopping | ✅ | ✅ |
| Learning rate scheduling | ❌ | ✅ |
| Solver options | ❌ | ✅ |
| Class weights | ❌ | ✅ |

## License

This implementation is provided as part of the BasicModels package.