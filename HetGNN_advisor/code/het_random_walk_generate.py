import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
import scipy.sparse as sp
import scipy.io as sio

def gen_het_rand_walk():
    a_a_list_train = [[] for k in range(7872)]
    f_name = '../data/academic/a_a_list_train.txt'
    neigh_f = open(f_name, "r")
    for line in neigh_f:
        line = line.strip()
        node_id = int(re.split(':', line)[0])
        neigh_list = re.split(':', line)[1]
        neigh_list_id = re.split(',', neigh_list)
        for j in range(len(neigh_list_id)):
            a_a_list_train[node_id].append('a' + str(neigh_list_id[j]))
        # elif f_name == 'p_a_list_train.txt':
        # 	for j in range(len(neigh_list_id)):
        # 		p_a_list_train[node_id].append('a'+str(neigh_list_id[j]))
        # elif f_name == 'p_p_citation_list.txt':
        # 	for j in range(len(neigh_list_id)):
        # 		p_p_cite_list_train[node_id].append('p'+str(neigh_list_id[j]))
        # else:
        # 	for j in range(len(neigh_list_id)):
        # 		v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
    neigh_f.close()

    het_walk_f = open("../data/academic/het_random_walk.txt", "w")
    # print len(self.p_neigh_list_train)
    for i in range(10):
        for j in range(7872):
            if len(a_a_list_train[j]):
                curNode = "a" + str(j)
                het_walk_f.write(curNode + " ")
                for l in range(20 - 1):
                    if curNode[0] == "a":
                        curNode = int(curNode[1:])
                        if(len(a_a_list_train[curNode])):
                            curNode_temp = random.choice(a_a_list_train[curNode])
                            if(len(a_a_list_train[int(curNode_temp[1:])])):
                                curNode = curNode_temp
                                het_walk_f.write(curNode + " ")
                            else:
                                curNode = "a" + str(curNode)
                                het_walk_f.write(curNode + " ")

                    # elif curNode[0] == "p":
                    #     curNode = int(curNode[1:])
                    #     curNode = random.choice(p_neigh_list_train[curNode])
                    #     het_walk_f.write(curNode + " ")
                    # elif curNode[0] == "v":
                    #     curNode = int(curNode[1:])
                    #     curNode = random.choice(v_p_list_train[curNode])
                    #     het_walk_f.write(curNode + " ")
                het_walk_f.write("\n")
    het_walk_f.close()

if __name__ == '__main__':
    gen_het_rand_walk()