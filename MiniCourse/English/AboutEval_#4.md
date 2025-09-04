# Mini Course on AI and Machine Learning

## Lesson 4: Evaluating Model Accuracy and Adding More Features

Welcome to Lesson 4! By now, you've learned the basics of AI and machine learning, prepared data, and built your first model with code. Today, we'll focus on making your models better: evaluating how well they perform and adding more features to improve predictions. Let's keep building on our house price example!

---

### Why Evaluate Model Accuracy?

Imagine baking cookies. You follow a recipe, but how do you know if they're any good? You taste them! In machine learning, evaluation is like tasting your cookies—it tells you how well your model is doing and where it can improve.

Without evaluation, you might think your model is great, but it could be making big mistakes. Evaluation helps you measure errors and make your model more reliable.

---

### Key Metrics for Evaluation

Here are simple ways to check your model's performance:

1. **Mean Absolute Error (MAE)**
   - **What is it?** The average of the absolute differences between predicted and actual values.
   - **Example:** If your model predicts $210,000 for a house that costs $200,000, the error is $10,000. MAE averages these errors.
   - **Analogy:** Like measuring how far off your guesses are in a game, on average.

2. **Mean Squared Error (MSE)**
   - **What is it?** Similar to MAE, but squares the errors before averaging (punishes big mistakes more).
   - **Example:** For the same $10,000 error, MSE would be (10,000)^2 = 100,000,000.
   - **Analogy:** Like scoring a game where big misses hurt your score more.

3. **R-Squared (R²)**
   - **What is it?** A score from 0 to 1 showing how well the model explains the data (1 is perfect).
   - **Example:** An R² of 0.8 means 80% of the price variation is explained by your model.
   - **Analogy:** Like a report card grade—higher is better!

---

### Step-by-Step: Evaluating Your House Price Model

Let's use the code from Lesson 3 and add evaluation.

#### 1. Import Libraries and Prepare Data

We'll use the same data, plus add a test set.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Our data (expanded)
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400],
    'Price': [200000, 250000, 220000, 300000, 180000, 240000]
}
df = pd.DataFrame(data)

# Split into training and testing sets
X = df[['Size']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 2. Train the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 3. Make Predictions and Evaluate

```python
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"Mean Squared Error: ${mse:,.0f}")
print(f"R-Squared: {r2:.2f}")
```

#### 4. Interpret the Results

- Low MAE/MSE and high R² mean a good model.
- If errors are high, your model needs improvement (more data, better features).

---

### Adding More Features

Our model only uses house size. Let's add location to make it smarter!

#### Why Add Features?

More features give the model more information, like adding spices to a recipe for better flavor.

#### Updated Data with Location

Let's encode location as numbers (City=1, Suburb=0).

```python
# Updated data
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400],
    'Location': [1, 0, 1, 0, 1, 0],  # 1=City, 0=Suburb
    'Price': [200000, 250000, 220000, 300000, 180000, 240000]
}
df = pd.DataFrame(data)

X = df[['Size', 'Location']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE with Location: ${mae:,.0f}")
print(f"R² with Location: {r2:.2f}")
```

#### What Changed?

- The model now considers both size and location.
- Predictions should be more accurate (lower MAE, higher R²).

---

### Common Pitfalls and Tips

- **Overfitting:** Model is too good on training data but bad on new data. Solution: Use more data or simpler models.
- **Underfitting:** Model is too simple and misses patterns. Solution: Add features or use better models.
- **Tip:** Always split data into train/test sets to avoid overfitting.

---

### Key Takeaways

- Evaluate models with metrics like MAE, MSE, and R² to measure accuracy.
- Adding features (like location) can improve predictions.
- Evaluation is key to building reliable AI models.

---

### Challenge: Try It Yourself!

- Add more features to the house data (e.g., number of bedrooms).
- Experiment with different test sizes (e.g., 20% vs. 50%).
- Compare MAE before and after adding features.

---

### What’s Next?

In the next lesson, we'll explore different types of models (beyond linear regression) and when to use them. Keep practicing and see you soon!
