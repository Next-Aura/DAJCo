# Mini Course on AI and Machine Learning

## Lesson 5: Exploring Different Machine Learning Models

Welcome to Lesson 5! By now, you've built and evaluated your first machine learning model using linear regression. But linear regression is just one tool in the toolbox. In this lesson, we'll explore other popular models, understand when to use them, and see how they compare. We'll keep using our house price example to make it relatable. Let's dive in and expand your modeling skills!

---

### Why Explore Different Models?

Imagine you're a chef with only one knife. You can cook, but some tasks would be easier with a different tool—like a whisk for beating eggs or a grater for cheese. Similarly, different machine learning models are better suited for different problems. Linear regression works well for straight-line relationships, but real-world data can be more complex, with curves, categories, or non-linear patterns.

Choosing the right model can make your predictions more accurate and your AI more powerful. We'll cover a few key models and when to pick them.

---

### Key Machine Learning Models

Here are some common models beyond linear regression, explained simply with analogies and examples.

1. **Decision Trees**
   - **What is it?** A model that makes decisions by asking yes/no questions, like a flowchart.
   - **Analogy:** Like a game of 20 Questions—starting with broad questions (e.g., "Is the house size > 1200 sq ft?") and narrowing down to a prediction.
   - **When to use:** For classification (e.g., yes/no decisions) or regression with non-linear data. Great for interpretable results.
   - **Pros:** Easy to understand and visualize.
   - **Cons:** Can overfit if too deep.

2. **Random Forest**
   - **What is it?** A group of decision trees working together, like a team of experts voting on a decision.
   - **Analogy:** Like asking multiple friends for advice on house prices—they each give an opinion, and you average them for a better answer.
   - **When to use:** For both classification and regression, especially when you want robustness and reduced overfitting.
   - **Pros:** Handles complex data well and is less prone to errors.
   - **Cons:** Slower to train and harder to interpret.

3. **Support Vector Machines (SVM)**
   - **What is it?** A model that finds the best "line" (or hyperplane) to separate data points.
   - **Analogy:** Like drawing the widest possible road between two groups of houses (e.g., expensive vs. cheap) without crossing into the wrong side.
   - **When to use:** For classification with clear boundaries, or regression. Works well with smaller datasets.
   - **Pros:** Effective in high-dimensional spaces.
   - **Cons:** Can be slow on large datasets and sensitive to parameters.

4. **K-Nearest Neighbors (KNN)**
   - **What is it?** A lazy model that predicts based on the "neighbors" of a data point.
   - **Analogy:** Like asking your closest neighbors what their house prices are to estimate yours.
   - **When to use:** For simple classification or regression, especially with small datasets.
   - **Pros:** No training phase, just stores data.
   - **Cons:** Slow for predictions on large data and sensitive to irrelevant features.

---

### Step-by-Step: Trying Different Models on House Prices

Let's compare linear regression with decision trees and random forest using our house price data. We'll use scikit-learn for the code.

#### 1. Import Libraries and Prepare Data

We'll expand our data with more features for better comparison.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Expanded house data
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400, 1600, 1100],
    'Bedrooms': [2, 3, 3, 4, 2, 3, 4, 2],
    'Location': [1, 0, 1, 0, 1, 0, 0, 1],  # 1=City, 0=Suburb
    'Price': [200000, 250000, 220000, 300000, 180000, 240000, 280000, 210000]
}
df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms', 'Location']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 2. Train and Evaluate Linear Regression (Review)

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"Linear Regression - MAE: ${lr_mae:,.0f}, R²: {lr_r2:.2f}")
```

#### 3. Train and Evaluate Decision Tree

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

print(f"Decision Tree - MAE: ${dt_mae:,.0f}, R²: {dt_r2:.2f}")
```

#### 4. Train and Evaluate Random Forest

```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest - MAE: ${rf_mae:,.0f}, R²: {rf_r2:.2f}")
```

#### 5. Compare the Results

Run the code and compare the MAE and R² values. Which model performs best? Random Forest often wins due to its ensemble nature, but results depend on your data.

---

### Choosing the Right Model

- **Start Simple:** Begin with linear regression for quick baselines.
- **For Accuracy:** Try random forest or SVM for complex data.
- **For Speed:** Use KNN or decision trees for smaller datasets.
- **Tip:** Always experiment and evaluate—there's no one-size-fits-all model.

---

### Common Pitfalls and Tips

- **Overfitting:** Complex models like deep trees can memorize training data. Use cross-validation to check.
- **Underfitting:** Simple models miss patterns. Add features or try different models.
- **Hyperparameters:** Tune settings (e.g., tree depth) for better performance.
- **Tip:** Use libraries like scikit-learn's GridSearchCV to automate tuning.

---

### Key Takeaways

- Different models suit different problems—choose based on data complexity and goals.
- Ensemble models like random forest often provide better accuracy.
- Always compare models using evaluation metrics to pick the best one.
- Experimentation is key to finding the right tool for the job.

---

### Challenge: Try It Yourself!

- Add more data points or features (e.g., age of house) and re-run the models.
- Try SVM or KNN on the house data using scikit-learn.
- Visualize a decision tree using `plot_tree` from sklearn.
- Compare performance on a classification task (e.g., predict if price > $250k).

---

### What’s Next?

In the next lesson, we'll dive into deploying your models—turning them from code into real-world applications. You'll learn how to save, load, and use models in production. Keep exploring and see you soon!
