import tensorflow as tf
import numpy as np

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        # x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        # res = tf.matmul(x, y, a_is_sparse=True)
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def weight_variable_same(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    #init_range = np.sqrt(1.0 / (input_dim + output_dim))
    initial = tf.ones([input_dim, output_dim], dtype=tf.float32)
    #initial = tf.random_uniform([input_dim, output_dim], minval=init_range,maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def weight_variable_zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def weight_variable_ones(shape, name=None):
    """All zeros."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def generate_degree_matrix(matrix):
    degree = np.zeros(705)
    colsum = matrix.sum(axis=0)
    rowsum = matrix.sum(axis=1)
    for j in range(0, 705):
        degree[j] = colsum[0, j] + rowsum[j, 0]
    A = matrix.diagonal()
    d = A.flat
    diagMat = list(d)
    return np.diag(degree - diagMat)