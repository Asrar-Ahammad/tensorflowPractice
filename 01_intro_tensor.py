import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print(tf.__version__)

# Basic operation on tensors
# Initializing a tensor
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
print(x)

# Matrices
x = tf.ones((3, 3))
print(x)
x = tf.zeros((2, 3))
print(x)
x = tf.eye(3)
print(x)

# For generating distributions
x = tf.random.normal((3, 3), mean=0, stddev=1)
print(x)
x = tf.random.uniform((1, 3), minval=0, maxval=1)
print(x)
x = tf.range(start=1, limit=10, delta=2)
print(x)

# Converting datatypes
x = tf.cast(x, dtype=tf.float64)
print(x)

# Mathematicl operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

# ADD
print(tf.add(x, y))
print(x + y)
# Subtract
print(tf.subtract(x, y))
print(x - y)
# Multiply
print(tf.multiply(x, y))
print(x * y)
# Division
print(tf.divide(x, y))
print(x / y)

# Matrix multiplication
x = tf.random.normal((1, 3))
y = tf.random.normal((3, 1))
print(x @ y)
print(tf.matmul(x, y))

# Indexing of tensors
x = tf.constant([2, 3, 4, 5, 1, 6, 7, 2, 4])
print(x[:])
print(x[:3])
print(x[::2])
print(x[::-1])

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)
x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
print(x[0, :])
print(x[0:2, :])

# Reshaping tensor
x = tf.range(9)
print(x)
x = tf.reshape(x, (3, 3))
print(x)
x = tf.transpose(x, perm=[1,0])
print(x)
