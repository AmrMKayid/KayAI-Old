import numpy as np

np.random.seed(7)

print('\n## Element-wise operations\n')

def relu_naive(x):
	assert len(x.shape) == 2	# x is a 2D Numpy tensor.

	x = x.copy()				# Avoid overwriting the input tensor.
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i, j] = max(x[i, j], 0)

	return x


def naive_add(x, y):
	assert len(x.shape) == 2
	assert x.shape == y.shape

	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i, j] += y[i, j] 

	return x


x = np.array([[-5, 78, 2, -34, 0],
          [6, 79, -3, 35, 1],
          [7, -80, 4, 36, -2]])

y = np.array([[7, 80, 4, 36, 2],
          [5, 78, 2, 34, 0],
          [6, 79, 3, 35, 1]])

print('\nNaive Non-Vectorized Implementation\n')
print('Relu naive', relu_naive(x))

print('\n x + y =', naive_add(x, y))

print('\nOptimized Basic Linear Algebra Subprograms (BLAS) implementation\n')

z = x + y
print(z)
z = np.maximum(z, 0.)
print(z)


print('\n## Broadcasting\n')

def naive_add_matrix_and_vector(x, y):
	assert len(x.shape) == 2
	assert len(y.shape) == 1
	assert x.shape[1] == y.shape[0]

	x = x.copy()
	for i in range(x.shape[0]):
	    for j in range(x.shape[1]):
	        x[i, j] += y[j]

	return x


x = np.random.random((10, 5))
y = np.random.random(5)
print('Matrix + vector =>', naive_add_matrix_and_vector(x, y))


x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)
print('\nShape of Z: ', z.shape)


print('\n## Tensor dot\n')

def naive_vector_dot(x, y):
	assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
		z += x[i] * y[i]		

	return z


def naive_matrix_vector_dot(x, y):
	assert len(x.shape) == 2
	assert len(y.shape) == 1
	assert x.shape[1] == y.shape[0]

	z = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
	
	return z


def naive_matrix_vector_dot2(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
    	z[i] = naive_vector_dot(x[i, :], y)

    return z    	


def naive_matrix_dot(x, y):
	assert len(x.shape) == 2
	assert len(y.shape) == 2
	assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			row_x = x[i, :]
			column_y = y[:, j]
			z[i, j] = naive_vector_dot(row_x, column_y)

	return z











































