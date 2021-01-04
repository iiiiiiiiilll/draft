import numpy as np


def rank(X):
    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for k in range(X.shape[2]):
            a = [X[i, j, k] for j in range(X.shape[1])]
            for j in range(X.shape[1]):
                if np.isnan(a[j]):
                    result[i, j, k] = np.nan
                else:
                    result[i, j, k] = sum(
                        [1 if (not np.isnan(a[l]) and (a[l] < a[j] or (a[l] == a[j] and l < j))) else 0
                         for l in range(X.shape[1])]) + 1

    return result


y = np.random.randint(1, 10, size=(2, 4, 10))
x = np.where(np.random.binomial(1, 0.7, size=y.shape), y, np.nan)
print(x)
print(rank(x))
