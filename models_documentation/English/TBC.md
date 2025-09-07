# TBC – Tree Backend Classifier

## Overview
TBC (Tree Backend Classifier) is a custom decision tree classifier implementation that supports various impurity criteria including Gini impurity, entropy, and log loss. It uses recursive binary splitting to build classification trees and includes support for sparse matrices with optional Numba JIT compilation for performance.

## Installation & Requirements
```bash
pip install numpy scipy numba
```

## Mathematical Formulation

### Decision Tree Classification
The model builds a binary decision tree by recursively splitting the feature space:

**Prediction**:
$$\hat{y} = \arg\max_c \left( \frac{1}{|L|} \sum_{i \in L} \mathbb{I}(y_i = c) \right)$$

Where $L$ is the set of training samples in the leaf node and $c$ represents class labels.

### Impurity Criteria

**Gini Impurity**:
$$I_{Gini} = 1 - \sum_{c=1}^{C} p_c^2$$

Where $p_c$ is the proportion of samples of class $c$ in the node.

**Entropy**:
$$I_{Entropy} = -\sum_{c=1}^{C} p_c \log_2(p_c)$$

**Log Loss (Cross-Entropy)**:
$$I_{LogLoss} = -\sum_{c=1}^{C} p_c \log(p_c)$$

### Split Quality
**Impurity Decrease (Gain)**:
$$Gain = I_{parent} - \frac{|S_{left}|}{|S|} I_{left} - \frac{|S_{right}|}{|S|} I_{right}$$

## Key Features
- ✅ Supports Gini, Entropy, and Log Loss criteria
- ✅ Works with dense and sparse matrix inputs
- ✅ Optional Numba JIT compilation for performance
- ✅ Comprehensive input validation with warnings
- ✅ Recursive tree building with configurable stopping criteria
- ✅ Multi-class classification support
- ✅ Variance-based feature selection for sparse matrices

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int \| None` | `2` | Maximum depth of the tree. If None or <=0, defaults to 2 |
| `min_samples_leaf` | `int \| None` | `5` | Minimum number of samples required to be at a leaf node |
| `criterion` | `Literal['gini', 'entropy', 'log_loss'] \| None` | `'gini'` | Function to measure the quality of a split |
| `min_samples_split` | `int \| None` | `2` | Minimum number of samples required to split an internal node |
| `min_impurity_decrease` | `float \| None` | `0.0` | Minimum impurity decrease required for a split |

## Model Attributes (After Fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `tree` | `dict` | Nested dictionary representing the tree structure |
| `max_depth` | `int` | Maximum depth of the tree |
| `min_samples_leaf` | `int` | Minimum samples per leaf |
| `criterion` | `str` | Impurity criterion used |
| `min_samples_split` | `int` | Minimum samples required to split |
| `min_impurity_decrease` | `float` | Minimum impurity decrease for splits |

## API Reference

### `TBClassifier.__init__()`
Initialize the tree classifier with hyperparameters.

### `TBClassifier.fit(X, y)`
Build the decision tree classifier from the training set.

**Parameters**:
- `X`: Training features (n_samples, n_features), dense or sparse
- `y`: Target class labels (n_samples,)

**Returns**: self

**Raises**:
- `ValueError`: If X and y have mismatched lengths or contain invalid data

### `TBClassifier.predict(X)`
Predict class labels for X.

**Parameters**:
- `X`: Test features (n_samples, n_features), dense or sparse

**Returns**: (n_samples,) predicted class labels

## Usage Examples

### Basic Decision Tree Classification
```python
from BasicModels.TBC import TBClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = TBClassifier(
    max_depth=5,
    min_samples_leaf=10,
    criterion='gini',
    min_samples_split=20
)
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate
train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Using Different Criteria
```python
from BasicModels.TBC import TBClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)

# Gini criterion
model_gini = TBClassifier(criterion='gini', max_depth=4)
model_gini.fit(X, y)
print(f"Gini Accuracy: {np.mean(model_gini.predict(X) == y):.4f}")

# Entropy criterion
model_entropy = TBClassifier(criterion='entropy', max_depth=4)
model_entropy.fit(X, y)
print(f"Entropy Accuracy: {np.mean(model_entropy.predict(X) == y):.4f}")

# Log Loss criterion
model_logloss = TBClassifier(criterion='log_loss', max_depth=4)
model_logloss.fit(X, y)
print(f"Log Loss Accuracy: {np.mean(model_logloss.predict(X) == y):.4f}")
```

### Multi-class Classification
```python
from BasicModels.TBC import TBClassifier
from sklearn.datasets import make_classification
import numpy as np

# Generate multi-class data
X, y = make_classification(
    n_samples=800,
    n_features=8,
    n_classes=4,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

model = TBClassifier(
    max_depth=6,
    min_samples_leaf=15,
    criterion='entropy'
)
model.fit(X, y)

predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Multi-class Accuracy: {accuracy:.4f}")
print(f"Unique classes: {np.unique(y)}")
```

### Sparse Matrix Support
```python
from BasicModels.TBC import TBClassifier
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse data
X_dense = np.random.randn(300, 15)
X_sparse = csr_matrix(X_dense)
y = (X_dense[:, 0] + X_dense[:, 1] > 0).astype(int)

model = TBClassifier(max_depth=4, criterion='gini')
model.fit(X_sparse, y)

print("Sparse matrix training completed successfully!")
accuracy = np.mean(model.predict(X_sparse) == y)
print(f"Accuracy: {accuracy:.4f}")
```

### Handling Parameter Warnings
```python
from BasicModels.TBC import TBClassifier
import warnings

# This will trigger warnings and use defaults
model = TBClassifier(
    max_depth=-1,  # Invalid, will warn and set to 2
    min_samples_leaf=0,  # Invalid, will warn and set to 5
    criterion='gini'
)

# Warnings will be shown:
# "max_depth should be a positive integer. Setting to default value of 2."
# "min_samples_leaf should be a positive integer. Setting to default value of 5."

print(f"Actual max_depth: {model.max_depth}")
print(f"Actual min_samples_leaf: {model.min_samples_leaf}")
```

## Best Practices

### Hyperparameter Tuning
- Start with `max_depth=3-5` and increase for complex datasets
- Use `min_samples_leaf=5-10` to prevent overfitting
- For large datasets, increase `min_samples_split` to 20-50
- Set `min_impurity_decrease > 0` to prune weak splits
- Choose criterion based on dataset: 'gini' for speed, 'entropy' for information gain

### Data Preprocessing
- No scaling required for decision trees
- Handle categorical features by encoding to numeric
- Consider feature engineering for better splits
- Remove or impute missing values
- Ensure class labels are integers starting from 0

### Tree Visualization
```python
# Simple tree structure inspection
def print_tree(node, depth=0):
    indent = "  " * depth
    if "label" in node:
        print(f"{indent}Leaf: Class {node['label']}")
    else:
        print(f"{indent}Split on feature {node['feature']} <= {node['value']:.4f}")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

print_tree(model.tree)
```

## Error Handling

The model includes comprehensive error checking:
- Input validation for empty arrays
- Shape consistency checks between X and y
- Parameter validation with informative warnings
- Criterion validation
- Sparse matrix compatibility checks
- Handling of edge cases (single class, no valid splits)

## Performance Considerations

- **Memory**: Efficient with both dense and sparse matrices
- **Speed**: Numba JIT acceleration for impurity calculations
- **Scalability**: Suitable for datasets up to 100k samples
- **Multi-class**: Handles multiple classes efficiently

## Comparison with scikit-learn

| Feature | TBC | scikit-learn DecisionTreeClassifier |
|---------|------|-------------------------------------|
| Custom criteria | ✅ | ✅ |
| Sparse matrix support | ✅ | ✅ |
| Numba acceleration | ✅ | ❌ |
| Parameter warnings | ✅ | ❌ |
| Tree pruning | ✅ | ✅ |
| Cost complexity pruning | ❌ | ✅ |
| Multi-output | ❌ | ✅ |
| Class weights | ❌ | ✅ |

## License

This implementation is provided as part of the BasicModels package.
