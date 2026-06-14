import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance plot
plt.barh(iris.feature_names, rf.feature_importances_)
plt.xlabel("Importance"); plt.title("Feature Importances"); plt.tight_layout(); plt.show()

# PCA visualization
X_pca = PCA(n_components=2).fit_transform(X_test)
for i, name in enumerate(iris.target_names):
    mask = y_test == i
    plt.scatter(X_pca[mask,0], X_pca[mask,1], label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of Test Set"); plt.legend(); plt.show()