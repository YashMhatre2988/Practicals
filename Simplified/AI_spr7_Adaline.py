import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([0,0,0,1])
w = np.zeros(2); b=0; lr=0.1

print(f"Initial Weights:{w} Bias:{b}")

for ep in range(10):
    print(f"\nEpoch {ep+1}")
    for x,t in zip(X,T):
        yin = b + np.dot(w,x)
        e = t - yin
        w += lr*e*x; b += lr*e
        print(f"Input:{x} Target:{t} Output:{yin:.2f}")
        print(f"Weights:{w} Bias:{round(b,2)}")

print("\nTesting Adaline (AND):")
for x in X:
    print(f"{x} -> {1 if b+np.dot(w,x)>=0.5 else 0}")