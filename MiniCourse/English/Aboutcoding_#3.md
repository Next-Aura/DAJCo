# Mini Course on AI and Machine Learning

## Lesson 3: Your First Machine Learning Code – Predicting House Prices

Welcome to Lesson 3! Now that you know what AI is and how to prepare data, let’s get hands-on and build your first machine learning model using code. Don’t worry if you’re new to programming—this guide is step-by-step and beginner-friendly.

---

### Why Python?

Python is the most popular language for AI and machine learning. It’s like the “universal remote” for data science: simple, powerful, and lots of helpful libraries.

---

### Setting Up Your Tools

Before we start, you’ll need:
- **Python** (download from [python.org](https://www.python.org/))
- **Jupyter Notebook** (lets you write and run code in your browser)
- **scikit-learn** (a library for machine learning)

**Quick Setup (in your terminal or command prompt):**
```sh
pip install notebook scikit-learn pandas
```
Then start Jupyter with:
```sh
jupyter notebook
```

---

### Step-by-Step: Predicting House Prices

Let’s use the house price example from before. We’ll use a simple model called **Linear Regression**.

#### 1. Import Libraries

Think of libraries as toolkits. We’ll use:
- `pandas` for handling data (like Excel for Python)
- `scikit-learn` for machine learning

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
```

#### 2. Prepare the Data

Let’s create a small table of house data.

```python
# Our data
data = {
    'Size': [1000, 1500, 1200],
    'Price': [200000, 250000, 220000]
}
df = pd.DataFrame(data)
```

#### 3. Split Features and Labels

- **Features:** What the model uses to predict (Size)
- **Labels:** What we want to predict (Price)

```python
X = df[['Size']]  # Features (must be 2D)
y = df['Price']   # Labels
```

#### 4. Train the Model

Let’s teach the model using our data.

```python
model = LinearRegression()
model.fit(X, y)
```

#### 5. Make Predictions

Now, let’s predict the price of a new house (e.g., 1300 sq ft):

```python
predicted_price = model.predict([[1300]])
print(f"Predicted price for 1300 sq ft: ${predicted_price[0]:,.0f}")
```

---

### What’s Happening Here?

- **We gave the model examples** (sizes and prices).
- **The model learned the pattern** (bigger houses cost more).
- **We asked it to predict** the price for a new size.

#### Analogy

It’s like showing a friend a few house prices, then asking, “What do you think a 1300 sq ft house would cost?” The model makes its best guess based on what it learned.

---

### Key Takeaways

- Python and scikit-learn make machine learning accessible for everyone.
- You can build a simple model in just a few lines of code.
- The process: Prepare data → Train model → Make predictions.

---

### Challenge: Try It Yourself!

- Change the house sizes and prices in the data.
- Predict prices for other house sizes.
- What happens if you add more data?

---

### What’s Next?

In the next lesson, we’ll explore how to evaluate your model’s accuracy and introduce more features (like location). Keep experimenting and have fun!