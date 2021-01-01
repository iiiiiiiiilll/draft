import numpy as np
import itertools

array = [np.random.randint(10, size=(2, 3, i + 3)) for i in range(3)]
a = itertools.product(*([i.shape, [None] * len(i.shape)] for i in array))
axis = -1
mask = [True] * array[0].ndim
mask[axis] = False
for tensor_shape in a:
    result_shape = [0 for i in range(len(mask))]
    for i in range(len(mask)):
        for j in tensor_shape:
            if mask[i]:
                if j[i] is not None:
                    result_shape[i] = j[i]
                    break
                result_shape[i] = None
            else:
                if j[i] is None:
                    result_shape[i] = None
                    break
                result_shape[i] += j[i]
    for i in tensor_shape:
        print(i)
    print(result_shape)
    c = np.array(tensor_shape)
    c = c[:, -1]
    test = c.sum() if [tensor_shape[i][axis] is not None for i in
                       range(len(array))] == [True] * len(array) else None
    print(test)
    print('')
