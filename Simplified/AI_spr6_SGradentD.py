import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
w = np.zeros(2); b=0; lr=0.1

act = lambda z: 1 if z>=0 else 0

for ep in range(10):
    print(f"\nEpoch {ep+1}")
    for i in range(len(X)):
        e = y[i] - act(np.dot(X[i],w)+b)
        w += lr*e*X[i]; b += lr*e
        print(f"Input:{X[i]} Pred:{act(np.dot(X[i],w)+b)} Error:{e}")

print(f"\nWeights:{w} Bias:{b}")

print("\nTesting:")
for xi in X:
    print(f"{xi} -> {act(np.dot(xi,w)+b)}")