
import tensorflow as tf
from layers import GraphConvolution
from decoders import BilinearDecoder,DistMultDecoder
from heteros import Heteros
from inits import *
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.hetero=None
        self.decoder=None
        self.inputs = None
        self.outputs = None

        self.loss = 0

        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.inputs=self.hetero(self.placeholders['hetero_feature'])
        # self.inputs = tf.eye(self.input_dim, dtype=tf.float32)
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self.predict()
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim,output_dim,hetero_input_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.hetero_input_dim=hetero_input_dim

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        one_relation=tf.ones_like(self.neg_relation,dtype=tf.float32)
        self.loss+=-tf.reduce_sum(tf.log(self.pos_relation))-tf.reduce_sum(tf.log(one_relation-self.neg_relation))

    def _adj_bias(self,edge_feature):
        node_size=edge_feature.get_shape().as_list()[0]
        w=tf.reshape(weight_variable_glorot(1,6,name='edge_weight'),shape=(1,6,1))
        self.edge_weight=tf.tile(w,[node_size,1,1])
        self.edge_bias=weight_variable_zeros([1],name='edge_bias')
        edge_press=tf.reshape(tf.matmul(edge_feature,self.edge_weight)+self.edge_bias,shape=(node_size,node_size))
        self.abias=edge_press
        self.loss+=tf.nn.l2_loss(w)+tf.nn.l2_loss(self.edge_bias)

    def _build(self):

        self._adj_bias(self.placeholders['edge_feature'])

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            abias=self.abias,  #使用了相似度计算出的链路信息权重
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            abias=self.abias,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            # sparse_inputs=True,
                                            logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=self.output_dim,
        #                                     placeholders=self.placeholders,
        #                                     abias=self.abias,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     # sparse_inputs=True,
        #                                     logging=self.logging))
        self.hetero=Heteros(self.hetero_input_dim,self.input_dim,self.placeholders,act=lambda x: x)
        self.decoder=BilinearDecoder(self.output_dim)

    def predict(self):
        row_edge_idx=self.placeholders['row_edge_idx']
        col_pos_edge_idx=self.placeholders['col_pos_edge_idx']
        col_neg_edge_idx=self.placeholders['col_neg_edge_idx']
        emb_row=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.outputs,row_edge_idx),1)
        emb_pos_col=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.outputs,col_pos_edge_idx),1)
        emb_neg_col=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.outputs,col_neg_edge_idx),1)
        print(self.outputs)
        self.pos_relation=self.decoder(emb_row,emb_pos_col)
        self.neg_relation=self.decoder(emb_row,emb_neg_col)
        return self.pos_relation,self.neg_relation
        

