
import scipy.sparse as sp
import scipy.io as sio
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

def user_user_matrix():
    node_ind=get_node_index()
    aut_num=len(node_ind)
    aut_aut_1=sp.lil_matrix((aut_num,aut_num),dtype=np.float32)
    aut_aut_2=sp.lil_matrix((aut_num,aut_num),dtype=np.float32)

    count1=0
    count2=0
    with open('refined_data/data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            if line[-1]=='1':
                aut_aut_1[id1,id2]=1.0
                aut_aut_1[id2,id1]=1.0
                count1+=1
            else:
                aut_aut_2[id1,id2]=1.0
                aut_aut_2[id2,id1]=1.0
                count2+=1
    size=aut_aut_1.shape[0]
    with open('refined_data/test_data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            if line[-1]=='1':
                aut_aut_1[id1,id2]=0.0
                aut_aut_1[id2,id1]=0.0
            else:
                aut_aut_2[id1,id2]=0.0
                aut_aut_2[id2,id1]=0.0
    user_user={
        'node_node_1':sp.csr_matrix(aut_aut_1),
        'node_node_2':sp.csr_matrix(aut_aut_2)
    }
    sio.savemat('refined_data/node_node.mat',user_user)


if __name__=="__main__":
    user_user_matrix()

