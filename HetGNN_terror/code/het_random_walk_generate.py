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
    a_a_list_train = [[] for k in range(1293)]
    f_name = '../data/academic/a_a_list_train.txt'
    neigh_f = open(f_name, "r")
    for line in neigh_f:
        line = line.strip()
        node_id = int(re.split(':', line)[0])
        neigh_list = re.split(':', line)[1]
        neigh_list_id = re.split(',', neigh_list)
        for j in range(len(neigh_list_id)):
            a_a_list_train[node_id].append('a' + str(neigh_list_id[j]))
    neigh_f.close()

    het_walk_f = open("../data/academic/het_random_walk.txt", "w")
    # print len(self.p_neigh_list_train)
    for i in range(10):
        for j in range(1293):
            if len(a_a_list_train[j]):
                curNode = "a" + str(j)
                het_walk_f.write(curNode + " ")
                for l in range(30 - 1):
                    if curNode[0] == "a":
                        curNode = int(curNode[1:])
                        curNode = random.choice(a_a_list_train[curNode])
                        het_walk_f.write(curNode + " ")

                het_walk_f.write("\n")
    het_walk_f.close()

if __name__ == '__main__':
    gen_het_rand_walk()