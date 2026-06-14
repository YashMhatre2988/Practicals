import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'Hours': range(1,11), 'Marks': [10,20,30,40,50,60,70,80,90,95]})
print("Dataset:\n", df)

h, m = np.array(df['Hours']), np.array(df['Marks'])
print(f"\nMean Marks: {np.mean(m):.2f}\nStd Dev: {np.std(m):.2f}")
print(f"Correlation: {stats.pearsonr(h, m)[0]:.4f}")

plt.scatter(df['Hours'], df['Marks'])
plt.title("Study Hours vs Marks"); plt.xlabel("Hours"); plt.ylabel("Marks")
plt.show()

X, y = df[['Hours']], df['Marks']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
model = LinearRegression().fit(Xtr, ytr)
print(f"\nPredicted: {model.predict(Xte)}\nActual: {list(yte)}")
print(f"Accuracy: {model.score(Xte, yte):.4f}")