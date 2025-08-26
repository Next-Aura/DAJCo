# GSBR – Gradient Supported Basic Regressor

## Overview
GSBR (Gradient Supported Basic Regressor) is a custom linear regression model that supports various regularization techniques including L1, L2, and ElasticNet. It uses mean squared error (MSE) or root mean squared error (RMSE) as loss functions, optimized using gradient descent.

## Key Features
- Supports L1, L2, and ElasticNet regularization
- Works with dense and sparse matrix inputs
- Loss functions: MSE or RMSE
- Includes early stopping and verbose training output

## Parameters
| Parameter       | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `penalty`       | Regularization type: 'l1', 'l2', 'elasticnet', None          |
| `alpha`         | Regularization strength                                      |
| `l1_ratio`      | Mix ratio for ElasticNet (only if `penalty` is 'elasticnet') |
| `loss`          | Loss function: 'mse' or 'rmse'                               |
| `fit_intercept` | Whether to include a bias term                               |

## Usage Example
```python
from BasicModels.GSBR import BasicRegressor
reg = BasicRegressor(penalty='elasticnet', loss='rmse', alpha=0.001)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```
