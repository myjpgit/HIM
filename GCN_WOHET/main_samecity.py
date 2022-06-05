
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from GCN_WOHET.models import GCN
from GCN_WOHET.minibatch import MiniBatch
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from GCN_WOHET.utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.seterr(invalid='ignore')


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'same_city', 'Dataset string.')
flags.DEFINE_integer('output_dim',64,'Output dimension.')
flags.DEFINE_integer('input_dim',512,'Output dimension.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoches',100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 10**-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('batch_size',100,'size of batch')

fea_att=extract_feature(FLAGS.dataset)
fea_att=sparse_to_tuple(fea_att)

node_node_1,node_node_2,node_fea,adj,efeature=extract_adj(FLAGS.dataset)
node_node_1=preprocess_adj(node_node_1)
node_node_2=preprocess_adj(node_node_2)
node_size=adj.shape[0]
FLAGS.input_dim=adj.shape[0]
node_fea=preprocess_features(node_fea)
#node_degree=preprocess_features(node_degree)
node_node_1=node_node_1.toarray()
node_node_2=node_node_2.toarray()


supports=[node_node_1,node_node_2]
adj+=np.eye(node_size)

hetero_input_dim=[fea_att[2][1]]
hetero_support=[node_fea]
hetero_feature=[fea_att]

# Define placeholders
placeholders = {
    'support': [tf.placeholder(tf.float32,shape=[node_size,node_size]) for _ in range(len(supports))],
    'hetero_support': [tf.sparse_placeholder(tf.float32) for _ in range(len(hetero_input_dim))],
    'hetero_feature': [tf.sparse_placeholder(tf.float32) for _ in range(len(hetero_input_dim))],
    'row_edge_idx':tf.placeholder(tf.int32,shape=[None]),
    'col_pos_edge_idx':tf.placeholder(tf.int32,shape=[None]),
    'col_neg_edge_idx':tf.placeholder(tf.int32,shape=[None]),
    'edge_feature':tf.placeholder(tf.float32,shape=[node_size,node_size,6])
}

def construct_feed_dict(row_edge_idx,col_pos_edge_idx,edge_feature,col_neg_edge_idx=None):
    feed_dict={}
    feed_dict.update({placeholders['support'][i]:supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['hetero_support'][i]:hetero_support[i] for i in range(len(hetero_support))})
    feed_dict.update({placeholders['hetero_feature'][i]: hetero_feature[i] for i in range(len(hetero_feature))})
    feed_dict.update({placeholders['row_edge_idx']:row_edge_idx}),
    feed_dict.update({placeholders['col_pos_edge_idx']:col_pos_edge_idx}),
    feed_dict.update({placeholders['edge_feature']:edge_feature})
    if not col_neg_edge_idx is None:
        feed_dict.update({placeholders['col_neg_edge_idx']:col_neg_edge_idx})
    return feed_dict

model=GCN(placeholders,FLAGS.input_dim,FLAGS.output_dim,hetero_input_dim)

res_auc=0.0
res_rank=0.0
res_F1=0.0
res_recall=0.0
res_pre=0.0
res_acc=0.0

for _ in range(10):
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    row_test,col_test=extract_test_data(FLAGS.dataset)
    label_test=np.repeat([1,0],len(row_test)//2)
    # row_valid, col_valid = extract_val_data(FLAGS.dataset)
    # label_valid = np.repeat([1, 0], len(row_valid) // 2)

    auc_test=0.0
    rank_test=0.0
    #precision_test=0.0
    #recall_test=0.0
    minibatch=MiniBatch(FLAGS.dataset,FLAGS.batch_size)

    for epoch in range(FLAGS.epoches):
        row_edge_idx,col_pos_edge_idx,col_neg_edge_idx=minibatch.get_batch_data()
        feed_dict=construct_feed_dict(row_edge_idx,col_pos_edge_idx,efeature,col_neg_edge_idx)
        outs=sess.run([model.opt_op,model.loss,model.abias],feed_dict=feed_dict)
        feed_dict = construct_feed_dict(row_test, col_test,efeature)
        test_pred = sess.run(model.pos_relation, feed_dict=feed_dict)
        auc_test = roc_auc_score(label_test, test_pred)
        test_pred = np.int64(test_pred >= 0.5)
        acc_test = accuracy_score(label_test, test_pred)
        precision_test = precision_score(label_test, test_pred, average='weighted')
        recall_test = recall_score(label_test, test_pred, average='weighted')
        F1_test = ((2.0 * precision_test * recall_test) / (precision_test + recall_test))
        print('iter: {},  acc of test: {:5f}, auc of test: {:5f}, F1_score of test: {:5f}, recall of test: {:5f}, Precision of test: {:5f}'.format(epoch,acc_test,auc_test,F1_test,recall_test,precision_test))
    #print(auc_test,rank_test)
    res_acc+=acc_test
    res_auc+=auc_test
    res_rank+=rank_test
    res_pre+=precision_test
    res_F1+=F1_test
    res_recall+=recall_test
    sess.close()
print('Auc:',res_auc/10, 'Acc:',res_acc/10, 'F1-score:',res_F1/10, 'Recall:',res_recall/10, 'Precision',res_pre/10)