inputs = [(0,0),(0,1),(1,0),(1,1)]
labels = [0,1,1,1]
w1=w2=b=0.0; lr=0.1

for _ in range(10):
    for (x1,x2),y in zip(inputs,labels):
        e = y - (1 if w1*x1+w2*x2+b >= 0 else 0)
        w1+=lr*e*x1; w2+=lr*e*x2; b+=lr*e

print(f"w1={w1}, w2={w2}, bias={b}")
for x1,x2 in inputs:
    print(f"{x1} OR {x2} = {1 if w1*x1+w2*x2+b>=0 else 0}")