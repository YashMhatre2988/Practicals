import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               VotingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Stochastic Gradient Boosting
sgb = GradientBoostingClassifier(n_estimators=100, subsample=0.8, random_state=42).fit(X_train, y_train)

# Voting Ensemble (soft voting across 3 diverse models)
vote = VotingClassifier(estimators=[
    ('rf',  RandomForestClassifier(n_estimators=50, random_state=42)),
    ('lr',  LogisticRegression(max_iter=200, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
], voting='soft').fit(X_train, y_train)

for name, model in [("AdaBoost", ada), ("Gradient Boosting", sgb), ("Voting Ensemble", vote)]:
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Accuracy comparison bar chart
names = ["AdaBoost", "Grad Boosting", "Voting"]
accs  = [accuracy_score(y_test, m.predict(X_test)) for m in [ada, sgb, vote]]
plt.bar(names, accs, color=['steelblue','darkorange','seagreen'])
plt.ylim(0.9, 1.01); plt.ylabel("Accuracy"); plt.title("Boosting & Voting Comparison")
plt.tight_layout(); plt.show()