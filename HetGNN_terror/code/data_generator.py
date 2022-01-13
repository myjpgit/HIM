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

class input_data(object):
    def __init__(self, args):
        self.args = args

        a_a_list_train = [[] for k in range(self.args.A_n)]

        relation_f = ["a_a_list_train.txt"]

        #store academic relational data
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.args.data_path + f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if f_name == 'a_a_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        a_a_list_train[node_id].append('a'+str(neigh_list_id[j]))

            neigh_f.close()

        self.a_a_list_train =  a_a_list_train


        if self.args.train_test_label != 2:
            self.triple_sample_p = self.compute_sample_p()

            #store pre-trained network/content embedding
            a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))

            net_e_f = sio.loadmat('../data/academic/node_feature.mat')
            a_net_embed = net_e_f['node_feature']
            a_net_embed = a_net_embed.A

            self.a_net_embed = a_net_embed

            #store neighbor set from random walk sequence
            a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(3)]

            het_neigh_train_f = open(self.args.data_path + "het_neigh.txt", "r")
            for line in het_neigh_train_f:
                line = line.strip()
                node_id = re.split(':', line)[0]
                neigh = re.split(':', line)[1]
                neigh_list = re.split(',', neigh)
                if node_id[0] == 'a' and len(node_id) > 1:
                    for j in range(len(neigh_list)):
                        if neigh_list[j][0] == 'a':
                            a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'p':
                            a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'v':
                            a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))

            het_neigh_train_f.close()

            #store top neighbor set (based on frequency) from random walk sequence
            a_neigh_list_train_top = [[[] for i in range(self.args.A_n)] for j in range(3)]
            top_k = [5, 5, 3] #fix each neighor type size
            for i in range(self.args.A_n):
                for j in range(1):
                    a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
                    top_list = a_neigh_list_train_temp.most_common(top_k[j])
                    #neigh_size = 0
                    neigh_size = 5
                    for k in range(len(top_list)):
                        a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                    if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
                        for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
                            a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

            a_neigh_list_train[:] = []

            self.a_neigh_list_train = a_neigh_list_train_top

            #self.gen_het_rand_walk()
            #store ids of author/paper/venue used in training
            train_id_list = [[] for i in range(2)]
            for i in range(1):
                if i == 0:
                    for l in range(self.args.A_n):
                        if len(a_neigh_list_train_top[i][l]):
                            train_id_list[i].append(l)
                    self.a_train_id_list = np.array(train_id_list[i])

    def compute_sample_p(self):
        print("computing sampling ratio for each kind of triple ...")
        window = self.args.window
        walk_L = self.args.walk_L
        A_n = self.args.A_n

        total_triple_n = [0.0] * 9 # nine kinds of triples
        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
        centerNode = ''
        neighNode = ''

        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[0] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[1] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[2] += 1
                    elif centerNode[0]=='p':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[3] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[4] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[5] += 1
                    elif centerNode[0]=='v':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[6] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[7] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[8] += 1
        het_walk_f.close()

        for i in range(1):
            total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
        print("sampling ratio computing finish.")

        return total_triple_n


    def sample_het_walk_triple(self):
        print ("sampling triple relations ...")
        triple_list = [[] for k in range(9)]
        window = self.args.window
        walk_L = self.args.walk_L
        A_n = self.args.A_n
        # P_n = self.args.P_n
        # V_n = self.args.V_n
        triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion

        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
                                    negNode = random.randint(0, A_n - 1)
                                    while len(self.a_a_list_train[negNode]) == 0:
                                        negNode = random.randint(0, A_n - 1)
                                    # random negative sampling get similar performance as noise distribution sampling
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[0].append(triple)
                    elif centerNode[0]=='p':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                    elif centerNode[0]=='v':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
        het_walk_f.close()

        return triple_list




