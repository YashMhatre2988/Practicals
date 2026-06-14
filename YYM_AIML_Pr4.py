import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Step 1: Create / Load Dataset
# ─────────────────────────────────────────────
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)


# ─────────────────────────────────────────────
# Step 2: NumPy Operations
# ─────────────────────────────────────────────
hours_array = np.array(df['Hours'])
marks_array = np.array(df['Marks'])

print("\nMean Marks (NumPy):", np.mean(marks_array))
print("Standard Deviation (NumPy):", np.std(marks_array))


# ─────────────────────────────────────────────
# Step 3: SciPy Statistical Analysis
# ─────────────────────────────────────────────
correlation, p_value = stats.pearsonr(hours_array, marks_array)
print("\nCorrelation (SciPy):", correlation)


# ─────────────────────────────────────────────
# Step 4: Data Visualization (Matplotlib)
# ─────────────────────────────────────────────
plt.scatter(df['Hours'], df['Marks'])
plt.title("Study Hours vs Marks")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()


# ─────────────────────────────────────────────
# Step 5: Machine Learning (Scikit-learn)
# ─────────────────────────────────────────────
X = df[['Hours']]   # Independent variable
y = df['Marks']     # Dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("\nPredicted Marks:", predictions)
print("Actual Marks:", list(y_test))

# Model Accuracy
score = model.score(X_test, y_test)
print("\nModel Accuracy:", score)
