
from collections import defaultdict
import random
from math import floor

def get_true_false():
    true=defaultdict(list)
    false=defaultdict(list)
    with open('data/relation_false.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            if id1<id2:
                false[id2].append(id1)
            else:
                false[id1].append(id2)
    with open('data/relation_true.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            if id1<id2:
                true[id2].append(id1)
            else:
                true[id1].append(id2)
    with open('refined_data/true_false.txt','w') as file:
        for i in true:
            file.write(str(i)+'\t'+str(true[i])+'\t'+str(false[i])+'\n')

def extract_true_false():
    relation={}
    with open('refined_data/true_false.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            relation[int(line[0])]=[eval(line[1]),eval(line[2])]
    return relation

def random_sample_advisee_collaborator(advisee_collaborator,idx,type):
    if len(advisee_collaborator[idx][type])>1:
        res=random.choice(advisee_collaborator[idx][type])
        # advisee_collaborator[idx][type].remove(res)
        return res
    else:
        return None

def get_val_test_data(val_num,test_num):
    false_true=extract_true_false()
    keys=list(false_true.keys())
    key_num=len(false_true)
    all_num=0
    for key,item in false_true.items():
        all_num+=len(item[0])
    print(floor(all_num*test_num))
    with open('refined_data/test_data.txt','w') as file:
        iter=0
        while iter<floor(all_num*test_num):
            key=keys[random.randint(0,key_num-1)]
            res=random_sample_advisee_collaborator(false_true,key,0)
            if res:
                iter+=1
                file.write(str(res)+'\t'+str(key)+'\t'+'1'+'\n')
                false_true[key][0].remove(res)

        iter=0
        while iter<floor(all_num*test_num):
            key=keys[random.randint(0,key_num-1)]
            res=random_sample_advisee_collaborator(false_true,key,1)
            if res:
                iter+=1
                file.write(str(res)+'\t'+str(key)+'\t'+'0'+'\n')
                false_true[key][1].remove(res)
    with open('refined_data/train_data.txt','w') as file:
        for adv,con in false_true.items():
            file.write(str(adv)+'\t'+str(con[0])+'\t'+str(con[1])+'\n')

    with open('data/train_data.txt','w') as file:
        for adv,con in false_true.items():
            for i in range(len(con[0])):
                file.write(str(adv) + '\t' + str(con[0][i]) + '\t' + '1' + '\n')
            for i in range(len(con[1])):
                file.write(str(adv) + '\t' + str(con[1][i]) + '\t' + '0' + '\n')

    with open('data/train_true_data.txt','w') as file:
        for adv,con in false_true.items():
            for i in range(len(con[0])):
                file.write(str(adv) + '\t' + str(con[0][i]) + '\t' + '1' + '\n')

def generate_data():
    relation=extract_true_false()
    with open('refined_data/data.txt','w') as file:
        for r in relation:
            for t in relation[r][0]:
                file.write(str(r)+'\t'+str(t)+'\t'+'1'+'\n')
            for f in relation[r][1]:
                file.write(str(r) + '\t' + str(f) + '\t' + '0' + '\n')

if __name__=='__main__':
    # get_true_false()
    get_val_test_data(50,0.6)
    # generate_data()