# TBR – Tree Backend Regressor

## Overview
TBR (Tree Backend Regressor) is a custom decision tree regressor implementation that supports various impurity criteria including MSE, Friedman MSE, Absolute Error, and Poisson deviance. It uses recursive binary splitting to build regression trees and includes support for sparse matrices with optional Numba JIT compilation for performance.

## Installation & Requirements
```bash
pip install numpy scipy numba
```

## Mathematical Formulation

### Decision Tree Regression
The model builds a binary decision tree by recursively splitting the feature space:

**Prediction**: 
$$\hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i$$

Where $L$ is the set of training samples in the leaf node.

### Impurity Criteria

**Mean Squared Error (MSE)**:
$$I_{MSE} = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

**Friedman MSE**:
$$I_{Friedman} = \frac{1}{|S|-1} \sum_{i \in S} (y_i - \bar{y})^2$$

**Mean Absolute Error**:
$$I_{MAE} = \frac{1}{|S|} \sum_{i \in S} |y_i - \bar{y}|$$

**Poisson Deviance**:
$$I_{Poisson} = 2 \sum_{i \in S} \left[ y_i \log\left(\frac{y_i}{\bar{y}}\right) - (y_i - \bar{y}) \right]$$

### Split Quality
**Impurity Decrease (Gain)**:
$$Gain = I_{parent} - \frac{|S_{left}|}{|S|} I_{left} - \frac{|S_{right}|}{|S|} I_{right}$$

## Key Features
- ✅ Supports MSE, Friedman MSE, Absolute Error, and Poisson criteria
- ✅ Works with dense and sparse matrix inputs
- ✅ Optional Numba JIT compilation for performance
- ✅ Comprehensive input validation
- ✅ Recursive tree building with configurable stopping criteria
- ✅ Feature subsampling support
- ✅ R² score calculation for evaluation

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `Optional[int]` | `5` | Maximum depth of the tree. If None, nodes are expanded until all leaves are pure |
| `min_samples_leaf` | `Optional[int]` | `1` | Minimum number of samples required to be at a leaf node |
| `criterion` | `Literal['mse', 'friedman_mse', 'absolute_error', 'poisson']` | `'mse'` | Function to measure the quality of a split |
| `max_features` | `Optional[int]` | `None` | Number of features to consider when looking for the best split |
| `random_state` | `Optional[int]` | `None` | Seed used by the random number generator |
| `min_samples_split` | `Optional[int]` | `2` | Minimum number of samples required to split an internal node |
| `min_impurity_decrease` | `Optional[float]` | `0.0` | Minimum impurity decrease required for a split |

## Model Attributes (After Fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `tree` | `dict` | Nested dictionary representing the tree structure |
| `max_depth` | `int` | Maximum depth of the tree |
| `min_samples_leaf` | `int` | Minimum samples per leaf |
| `criterion` | `str` | Impurity criterion used |
| `max_features` | `Optional[int]` | Number of features considered for splits |
| `random_state` | `Optional[int]` | Random state for reproducibility |
| `min_samples_split` | `int` | Minimum samples required to split |
| `min_impurity_decrease` | `float` | Minimum impurity decrease for splits |

## API Reference

### `TBRegressor.__init__()`
Initialize the tree regressor with hyperparameters.

### `TBRegressor.fit(X, y)`
Build the decision tree regressor from the training set.

**Parameters**:
- `X`: Training features (n_samples, n_features), dense or sparse
- `y`: Target values (n_samples,)

**Returns**: self

**Raises**:
- `ValueError`: If X and y have mismatched lengths or contain invalid data

### `TBRegressor.predict(X)`
Predict regression target for X.

**Parameters**:
- `X`: Test features (n_samples, n_features), dense or sparse

**Returns**: (n_samples,) predicted values

### `TBRegressor.score(X, y)`
Return the coefficient of determination R² of the prediction.

**Parameters**:
- `X`: Test features (n_samples, n_features)
- `y`: True target values (n_samples,)

**Returns**: R² score (float)

## Usage Examples

### Basic Decision Tree Regression
```python
from BasicModels.TBR import TBRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = TBRegressor(
    max_depth=10,
    min_samples_leaf=5,
    criterion='mse',
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
```

### Using Different Criteria
```python
from BasicModels.TBR import TBRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

# MSE criterion
model_mse = TBRegressor(criterion='mse', max_depth=8)
model_mse.fit(X, y)
print(f"MSE R²: {model_mse.score(X, y):.4f}")

# Friedman MSE
model_friedman = TBRegressor(criterion='friedman_mse', max_depth=8)
model_friedman.fit(X, y)
print(f"Friedman MSE R²: {model_friedman.score(X, y):.4f}")

# Absolute Error
model_mae = TBRegressor(criterion='absolute_error', max_depth=8)
model_mae.fit(X, y)
print(f"MAE R²: {model_mae.score(X, y):.4f}")
```

### Poisson Criterion for Count Data
```python
from BasicModels.TBR import TBRegressor
import numpy as np

# Generate Poisson-like data
np.random.seed(42)
X = np.random.randn(300, 4)
y = np.random.poisson(np.exp(X[:, 0] + 0.5 * X[:, 1]))  # Poisson target

model = TBRegressor(
    criterion='poisson',
    max_depth=6,
    min_samples_leaf=10
)
model.fit(X, y)

print(f"Poisson deviance R²: {model.score(X, y):.4f}")
```

### Sparse Matrix Support
```python
from BasicModels.TBR import TBRegressor
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse data
X_dense = np.random.randn(200, 20)
X_sparse = csr_matrix(X_dense)
y = X_dense[:, 0] + 0.5 * X_dense[:, 1] + np.random.randn(200) * 0.1

model = TBRegressor(max_depth=8, random_state=42)
model.fit(X_sparse, y)

print("Sparse matrix training completed successfully!")
print(f"R² score: {model.score(X_sparse, y):.4f}")
```

### Feature Subsampling
```python
from BasicModels.TBR import TBRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Use only 10 features for each split
model = TBRegressor(
    max_features=10,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

print(f"Random forest-like R²: {model.score(X, y):.4f}")
```

## Best Practices

### Hyperparameter Tuning
- Start with `max_depth=5-10` and increase for complex datasets
- Use `min_samples_leaf=1-5` to prevent overfitting
- For large datasets, increase `min_samples_split` to 10-20
- Set `min_impurity_decrease > 0` to prune weak splits
- Use `max_features` for ensemble-like behavior

### Data Preprocessing
- No scaling required for decision trees
- Handle categorical features by encoding to numeric
- Consider feature engineering for better splits
- Remove or impute missing values

### Tree Visualization
```python
# Simple tree structure inspection
def print_tree(node, depth=0):
    indent = "  " * depth
    if "value" in node:
        print(f"{indent}Leaf: {node['value']:.4f}")
    else:
        print(f"{indent}Split on feature {node['feature']} <= {node['threshold']:.4f}")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

print_tree(model.tree)
```

## Error Handling

The model includes comprehensive error checking:
- Input validation for NaN/Inf values
- Shape consistency checks between X and y
- Parameter validation (positive values, valid ranges)
- Criterion validation
- Sparse matrix compatibility checks

## Performance Considerations

- **Memory**: Efficient with both dense and sparse matrices
- **Speed**: Numba JIT acceleration for variance calculations
- **Scalability**: Suitable for datasets up to 100k samples
- **Parallelization**: Feature subsampling can be parallelized

## Comparison with scikit-learn

| Feature | TBR | scikit-learn DecisionTreeRegressor |
|---------|------|------------------------------------|
| Custom criteria | ✅ | ✅ |
| Sparse matrix support | ✅ | ✅ |
| Poisson criterion | ✅ | ❌ |
| Numba acceleration | ✅ | ❌ |
| Feature subsampling | ✅ | ✅ |
| Tree pruning | ✅ | ✅ |
| Cost complexity pruning | ❌ | ✅ |
| Multi-output | ❌ | ✅ |

## License

This implementation is provided as part of the BasicModels package.
