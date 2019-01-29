import numpy as np

print('\n## Scalars (0D tensors)')

x = np.array(12)
print(x)
print(x.ndim)
print(x.shape)

print('\n## Vectors (1D tensors)')

x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)
print(x.shape)

print('\n## Matrices (2D tensors)')
##### two axes (often referred to rows and columns)

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x)
print(x.ndim)
print(x.shape)

print('\n## 3D tensors and higher-dimensional tensors')

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
				[[5, 78, 2, 34, 0],
				 [6, 79, 3, 35, 1],
				 [7, 80, 4, 36, 2]],
				[[5, 78, 2, 34, 0],
				 [6, 79, 3, 35, 1],
				 [7, 80, 4, 36, 2]]])
print(x)
print(x.ndim)
print(x.shape)