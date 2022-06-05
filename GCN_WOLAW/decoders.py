
import tensorflow as tf
from GCN_WOLAW.inits import *


class BilinearDecoder():
    def __init__(self, input_dim, act=tf.nn.sigmoid, **kwargs):
        self.act = act
        self.vars={}
        self.name=self.__class__.__name__.lower()
        with tf.variable_scope(self.name + '_vars'):
                temp = weight_variable_glorot(
                    input_dim, input_dim, name='relation_weight')
                self.vars['relation_weight'] = temp #tf.reshape(tmp, [-1])

    def __call__(self, inputs_row, inputs_col):
        relation = self.vars['relation_weight']
        intermediate_product = tf.matmul(inputs_row, relation)
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        outputs=self.act(rec)
        return tf.diag_part(outputs)

class DistMultDecoder():
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, act=tf.nn.sigmoid, **kwargs):
        self.act = act
        self.vars={}
        self.name=self.__class__.__name__.lower()
        with tf.variable_scope(self.name + '_vars'):
                temp = weight_variable_glorot(
                    input_dim, 1, name='relation_weight')
                self.vars['relation_weight']=tf.reshape(temp, [-1])

    def __call__(self, inputs_row, inputs_col):
        relation = tf.diag(self.vars['relation_weight'])
        intermediate_product = tf.matmul(inputs_row, relation)
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        outputs=self.act(rec)
        return tf.diag_part(outputs)