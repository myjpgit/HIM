
import csv
from collections import defaultdict
import random
from math import floor

def to_int(li):
    return [int(x) for x in li]

def get_author_index():
    aut_ind={}
    count=0
    with open('refined_data/author_index.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            aut_ind[line]=count
            count+=1
    return aut_ind

def get_data():
    aut_ind=get_author_index()
    out_file=open('refined_data/data.txt','w')
    with open('refined_data/refined_true_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            out_file.write(str(aut_ind[line[0]])+'\t'+str(aut_ind[line[1]])+'\t'+'1'+'\n')
    with open('refined_data/cut_false_aa_by_diff_year.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            out_file.write(str(aut_ind[line[0]])+'\t'+str(aut_ind[line[1]])+'\t'+'0'+'\n')
    out_file.close()

def get_advisee_collaborator():
    advisees=defaultdict(list)
    collaborators=defaultdict(list)
    with open('refined_data/data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            if line[-1]=='1':
                advisees[line[1]].append(line[0])
            else:
                collaborators[line[1]].append(line[0])
    with open('refined_data/advisee_collaborator.txt','w') as file:
        for adv in advisees:
            file.write(adv+'\t'+str(advisees[adv])+'\t'+str(collaborators[adv])+'\n')

def get_advisor_collaborator():
    advisors=defaultdict(list)
    collaborators=defaultdict(list)
    with open('refined_data/data.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            if line[-1]=='1':
                advisors[line[0]].append(line[1])
            else:
                collaborators[line[0]].append(line[1])
    with open('refined_data/advisor_collaborator.txt','w') as file:
        for adv in advisors:
            file.write(adv+'\t'+str(advisors[adv])+'\t'+str(collaborators[adv])+'\n')

def extract_advisee_collaborator():
    advisee_collaborator={}
    with open('refined_data/advisee_collaborator.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            advisee_collaborator[int(line[0])]=[to_int(eval(line[1])),to_int(eval(line[2]))]
    return advisee_collaborator

def random_sample_advisee_collaborator(advisee_collaborator,idx,type):
    if len(advisee_collaborator[idx][type])>1:
        res=random.choice(advisee_collaborator[idx][type])
        # advisee_collaborator[idx][type].remove(res)
        return res
    else:
        return None

def get_val_test_data(val_num,test_num):
    advisee_collaborator=extract_advisee_collaborator()
    advisor_keys=list(advisee_collaborator.keys())
    author_num=len(advisee_collaborator)
    all_num=0
    for key,values in advisee_collaborator.items():
        all_num+=len(values[0])
    # with open('refined_data/val_data.txt','w') as file:
    #     iter=0
    #     while iter<val_num:
    #         key=advisor_keys[random.randint(0,author_num-1)]
    #         res=random_sample_advisee_collaborator(advisee_collaborator,key,0)
    #         if res:
    #             iter+=1
    #             file.write(str(res)+'\t'+str(key)+'\t'+'1'+'\n')
    #     iter=0
    #     while iter<val_num:
    #         key=advisor_keys[random.randint(0,author_num-1)]
    #         res=random_sample_advisee_collaborator(advisee_collaborator,key,1)
    #         if res:
    #             iter+=1
    #             file.write(str(res)+'\t'+str(key)+'\t'+'0'+'\n')
    with open('refined_data/test_data.txt','w') as file:
        iter=0
        while iter<floor(all_num*test_num):
            key=advisor_keys[random.randint(0,author_num-1)]
            res=random_sample_advisee_collaborator(advisee_collaborator,key,0)
            if res:
                iter+=1
                file.write(str(res)+'\t'+str(key)+'\t'+'1'+'\n')
                advisee_collaborator[key][0].remove(res)

        iter=0
        while iter<floor(all_num*test_num):
            key=advisor_keys[random.randint(0,author_num-1)]
            res=random_sample_advisee_collaborator(advisee_collaborator,key,1)
            if res:
                iter+=1
                file.write(str(res)+'\t'+str(key)+'\t'+'0'+'\n')
                advisee_collaborator[key][1].remove(res)
    with open('refined_data/train_data.txt','w') as file:
        for adv,con in advisee_collaborator.items():
            file.write(str(adv)+'\t'+str(con[0])+'\t'+str(con[1])+'\n')

    # with open('data/train_data.txt','w') as file:
    #     for adv,con in advisee_collaborator.items():
    #         for i in range(len(con[0])):
    #             file.write(str(adv) + '\t' + str(con[0][i]) + '\t' + '1' + '\n')
    #         for i in range(len(con[1])):
    #             file.write(str(adv) + '\t' + str(con[1][i]) + '\t' + '0' + '\n')

    with open('data/train_true_data.txt','w') as file:
        for adv,con in advisee_collaborator.items():
            for i in range(len(con[0])):
                file.write(str(adv) + '\t' + str(con[0][i]) + '\t' + '1' + '\n')


def extract_val_test_data(type):
    val_test_data=set()
    with open('refined_data/{}_data.txt'.format(type)) as file:
        for line in file:
            line=line.strip().split('\t')
            val_test_data.add((line[0],line[1]))
    return val_test_data

# def get_train_data():
#     val_data=extract_val_test_data('val')
#     test_data=extract_val_test_data('test')
#     val_test_data=val_data|test_data
#     out_file=open('refined_data/train_data.txt','w')
#     with open('refined_data/data.txt') as file:
#         for line in file:
#             con=line.strip().split('\t')
#             pair=(con[0],con[1])
#             if not pair in val_test_data:
#                 out_file.write(line)


if __name__=='__main__':
    # get_data()
    # get_advisee_collaborator()
    get_advisor_collaborator()
    get_val_test_data(500,0.1)
    # get_train_data()

