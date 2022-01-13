import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
import scipy.sparse as sp
import scipy.io as sio

end_node_embedding = [[] for k in range(10195)]
node_feature = sio.loadmat('../data/academic/node_feature.mat')
node_feature = node_feature['node_feature']
node_feature = node_feature.A
#print(node_feature[0])
f_name = '../data/academic/node_embedding.txt'
neigh_f = open(f_name, "r")
nodeid=[]
for line in neigh_f:
    line = line.strip()
    node_id = re.split(' ', line)[0]
    neigh_list = re.split(' ', line)[1:]
    for j in range(len(neigh_list)):
        end_node_embedding[int(node_id[1:])].append(float(neigh_list[j]))

neigh_f.close()

for k in range(10195):
    if (end_node_embedding[k]):
        end_node_embedding[k] = np.array(end_node_embedding[k])
    else:
        end_node_embedding[k] = node_feature[k]

print(end_node_embedding)
# file3 = open(r'../data/academic/end_node_embedding.txt', 'w',encoding='UTF-8')
# for i in range(len(end_node_embedding)):
#     file3.write(str(end_node_embedding[i])+'\n')
# file3.close()
end_node_embedding = sp.csr_matrix(end_node_embedding)
end_node_embedding={
        'node_feature':end_node_embedding
}
sio.savemat('../data/academic/end_node_embedding.mat', end_node_embedding)