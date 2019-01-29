from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('\n\n')

print('Tensor\'s key attributes:')
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)


digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()


print('\nManipulating tensors in Numpy\n')

my_slice = train_images[10:100]
print(my_slice.shape)

## Equivalent to the previous example
my_slice = train_images[10:100, :, :]
print(my_slice.shape)

## Equivalent to the previous example
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)


my_slice = train_images[:, 14:, 14:]
print(my_slice.shape)

my_slice = train_images[:, 7:-7, 7:-7]
print(my_slice.shape)


print('\nThe notion of data batches\n')

batch = train_images[:128]
batch = train_images[128:256]
# nth batch
n = 2
batch = train_images[128 * n:128 * (n + 1)]