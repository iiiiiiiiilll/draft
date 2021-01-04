import numpy as np
import itertools

y = np.random.randint(1, 10, size=(2, 4, 10))
X = np.where(np.random.binomial(1, 0.7, size=y.shape), y, np.nan)
axis = 1
print(X)

# 将axis换到最后一个位置
axis_ = axis if axis >= 0 else X.ndim + axis
T = list(a for a in range(X.ndim) if a != axis_) + [axis_]
rev_T = [0] * len(T)
for i, j in enumerate(T):
    rev_T[j] = i

X = X.transpose(list(a for a in range(X.ndim) if a != axis_) + [axis_])
front_shape = list(X.shape[:-1])
xx = X.reshape([-1, X.shape[-1]])
ans = np.zeros(xx.shape, dtype=xx.dtype)
for i in range(len(xx)):
    for j in range(xx.shape[1]):
        if np.isnan(xx[i, j]):
            ans[i, j] = np.nan
        else:
            ans[i, j] = sum(
                [
                    1 if (not np.isnan(xx[i, j])
                          and (
                                  xx[i, l] < xx[i, j] or (xx[i, l] == xx[i, j] and l < j)
                          )) else 0 for l in range(xx.shape[1])
                ]
            ) + 1

y = ans.reshape(front_shape + [xx.shape[-1]])
y = y.transpose(rev_T)

print(y)
