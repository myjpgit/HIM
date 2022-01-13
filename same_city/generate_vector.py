
import scipy.io as sio
import scipy.sparse as sp
import numpy as np

def get_node_index():
    node_ind={}
    count=0
    with open('refined_data/node_ind.txt') as file:
        for line in file:
            line=line.strip()
            node_ind[line]=count
            count+=1
    return node_ind

def node_feature_matrix():
    # node_ind=get_node_index()
    # node_num=len(node_ind)
    features=[]
    with open('refined_data/node_feature.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            fea=[float(x) for x in line[0:]]
            features.append(fea)
    features=sp.csr_matrix(features,dtype=np.float32)
    print(features)
    node_feature={
        'node_feature':features
    }
    sio.savemat('refined_data/node_feature.mat',node_feature)

def one_hot_vector():
    feature=sp.eye(150,dtype=np.float32)
    ff={
        'feature':sp.csr_matrix(feature)
    }
    sio.savemat('refined_data/one_hot_vector.mat',ff)

if __name__=='__main__':
    node_feature_matrix()
    #one_hot_vector()
    #one_hot_att_vector()




