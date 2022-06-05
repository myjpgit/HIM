
import tensorflow as tf
from GCN_WOHET.inits import *


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Heteros():
    def __init__(self,hetero_input_dim, output_dim, placeholders,act=tf.nn.relu, bias=False):

        self.act = act
        self.support = placeholders['support']
        self.hetero_support=placeholders['hetero_support']
        self.bias = bias
        self.hetero_feature=placeholders['hetero_feature']
        self.name=self.__class__.__name__.lower()
        self.vars={}
        self.output_dim=output_dim

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.hetero_support)):
                self.vars['weights_hetero_' + str(i)] = weight_variable_glorot(hetero_input_dim[i], output_dim,
                                                        name='weights_hetero_' + str(i))
            if self.bias:
                self.vars['bias'] = weight_variable_zeros([output_dim], name='bias')

    def __call__(self, input):
        x=input

        supports = list()
        for i in range(len(self.hetero_support)):
            pre_sup = dot(x[i], self.vars['weights_hetero_' + str(i)], sparse=True)
            pre = dot(self.hetero_support[i], pre_sup, sparse=True)
            support = dot(self.support[i], pre, sparse=False)
            supports.append(support)
        output = tf.add_n(supports)
        # output=tf.concat(supports,1)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

