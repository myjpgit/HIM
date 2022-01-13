from collections import defaultdict
import random
from math import floor

def get_true_false():
    true=defaultdict(list)
    false=defaultdict(list)
    with open('data/train_true_data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            if id1<id2:
                true[id1].append('a'+str(id2))
                true[id2].append('a' + str(id1))
            else:
                true[id2].append('a'+str(id1))
                true[id1].append('a' + str(id2))
    with open('../HetGNN_advisor/data/academic/het_neigh.txt','w') as file:
        for i in true:
            temp = 'a'+str(i)+':'+str(true[i])
            temp = temp.replace("[", "")
            temp = temp.replace("]", "")
            temp = temp.replace("'", "")
            temp = temp.replace(" ", "")
            file.write(temp +'\n')

if __name__=='__main__':
    true = get_true_false()
    print(true)
