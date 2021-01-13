from gplearn.genetic import *
from sklearn.utils.random import check_random_state
import numpy as np
from gplearn.functions import _Function, make_function

rng = check_random_state(0)


# def _logical(x1, x2, x3, x4):
#     return np.where(x1 > x2, x3, x4)
#
#
# logical = make_function(function=_logical,
#                         name='logical',
#                         arity=4)
# function_set = ['add', 'sub', 'mul', 'sin', logical]

# Training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
# y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] + X_train[:, 0] - 1
# y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + np.sin(X_train[:, 1]) - 1
# y_train = np.where(X_train[:, 0] > X_train[:, 1] + X_train[:, 2], X_train[:, 1], X_train[:, 3]) - X_train[:, 1]
# X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + np.sin(X_train[:, 1]) - 1


# sin这个例子，设置generations = 50, stopping_criteria=0.001时跑出了除常数项之外的所有项，在第38代停止
# 官方的例子里，若设置const_range=None, 在generation=2时就结束了，利用了div(X1, X1)来得出常数1

# 两列值代表两个自变量的值 X_train[:,0]=X_0, X_train[:,1]=X_1, 这里相当于给了50个(x,y)点
# Training samples 用来训练model,但是我们不能通过model在training sample上的表现来评价model,避免过拟合
# Testing samples
# X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
# y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

est_gp = SymbolicRegressor(population_size=5000,
                           # function_set=function_set,
                           # const_range=None,
                           # init_method='full',
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)
print(est_gp._program)
print(type(est_gp._program))
print(est_gp._program.parents)
print(est_gp._program.depth_)
print(est_gp._program.length_)
