
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
from math import log


def get_node_index():
    node_ind={}
    count=0
    with open('refined_data/node_ind.txt') as file:
        for line in file:
            line=line.strip()
            node_ind[line]=count
            count+=1
    return node_ind

def common_neighbor(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    return len(n1&n2)

def admic_adar(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    res=0.0
    if len(n1&n2)==0:
        return 0.0
    for k in n1&n2:
        if len(indices[k])>1:
            res+=1/log(len(indices[k]))
    return res

def jaccard(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    if len(n1)==0 or len(n2)==0:
        return 0.0
    return len(n1&n2)/len(n1|n2)

def edge_feature():
    node_ind=get_node_index()
    aut_num=len(node_ind)
    print(aut_num)
    aut_aut=sp.lil_matrix((aut_num,aut_num),dtype=np.float32)
    with open('refined_data/data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            aut_aut[id1,id2]=1
            aut_aut[id2,id1]=1
    aut_aut=sp.csr_matrix(aut_aut)
    degrees=np.ravel(np.sum(aut_aut,axis=1))
    indptr = aut_aut.indptr
    indices = aut_aut.indices
    split_indices = np.split(indices, indptr[1:-1])
    efeature=np.zeros((aut_num,aut_num,6),dtype=np.float32)
    sumf=np.zeros(shape=(6,),dtype=np.float32)
    for i in range(aut_num):
        fea = [degrees[i], degrees[i], common_neighbor(split_indices, i, i), \
                admic_adar(split_indices, i, i), jaccard(split_indices, i, i), degrees[i] * degrees[i]]
        # print(fea)
        efeature[i, i] = fea
        sumf += fea
        for j in split_indices[i]:
                fea=[degrees[i],degrees[j],common_neighbor(split_indices,i,j),\
                 admic_adar(split_indices,i,j),jaccard(split_indices,i,j),degrees[i]*degrees[j]]
                # print(fea)
                efeature[i,j]=fea
                sumf+=fea
    sumf=np.reciprocal(sumf)
    efeature=efeature*sumf
    print(np.max(efeature))
    np.save('refined_data/efeature',efeature)
    np.save('refined_data/adj',aut_aut.toarray())


if __name__=='__main__':
    edge_feature()