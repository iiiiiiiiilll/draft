import numpy as np

y = np.random.normal(10, 1, size=12)
x = np.random.randint(1, 10, size=12)
print(y)
print(x)
ans = np.array([])
for i in range(len(y)):
    z = np.array([])
    for j in range(len(x)):
        if x[j] == x[i]:
            z = np.append(z, y[j])
    ans = np.append(ans, y[i] - z.mean())
print(ans)
