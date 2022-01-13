
import scipy.sparse as sp
import scipy.io as sio
import numpy as np

def get_author_index():
    aut_ind={}
    count=0
    with open('refined_data/author_index.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            aut_ind[line]=count
            count+=1
    return aut_ind

def re_author_author_matrix():
    aut_ind=get_author_index()
    aut_num=len(aut_ind)
    aut_aut_1=sp.lil_matrix((aut_num,aut_num),dtype=np.float32)
    aut_aut_2=sp.lil_matrix((aut_num,aut_num),dtype=np.float32)

    with open('refined_data/train_data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            rela1=eval(line[1])
            rela2=eval(line[2])
            for r in rela1:
                aut_aut_1[id1,r]=1.0
                aut_aut_1[r, id1] = 1.0
            for r in rela2:
                aut_aut_2[id1,r]=1.0
                aut_aut_2[r, id1] = 1.0

    aut_aut={
        'author_author_1':sp.csr_matrix(aut_aut_1),
        'author_author_2':sp.csr_matrix(aut_aut_2)
    }
    sio.savemat('refined_data/author_author',aut_aut)

def re_author_author_matrix_2():
    aut_ind=get_author_index()
    aut_num=len(aut_ind)
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
    print(aut_aut_1.shape)
    print(count1)
    print(count2)
    print(count1+count2)
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
    aut_aut={
        'author_author_1':sp.csr_matrix(aut_aut_1),
        'author_author_2':sp.csr_matrix(aut_aut_2)
    }
    sio.savemat('refined_data/author_author',aut_aut)

if __name__=="__main__":
    re_author_author_matrix_2()

