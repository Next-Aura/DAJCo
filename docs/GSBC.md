# GSBC – Gradient Supported Basic Classifier

## Overview
GSBC (Gradient Supported Basic Classifier) is a custom-built classification model using logistic regression with gradient descent optimization. It supports both binary and multi-class classification using the One-vs-Rest (OvR) strategy.

## Key Features
- Supports binary and multi-class classification
- Handles dense and sparse matrices
- Uses Binary Cross Entropy Loss
- Optimized with Batch Gradient Descent
- Includes logging and early stopping features

## Parameters
| Parameter       | Description                              |
|-----------------|------------------------------------------|
| `max_iter`      | Maximum number of training iterations    |
| `learning_rate` | Step size for gradient descent           |
| `fit_intercept` | Whether to include a bias term           |
| `tol`           | Tolerance for loss change to stop early  |
| `verbose`       | Print training progress per epoch        |

## Usage Example
```python
from BasicModels.GSBC import BasicClassifier
model = BasicClassifier(max_iter=500, learning_rate=0.01, verbose=1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```
