
from collections import  defaultdict
'''useless'''
train_true=defaultdict(list)
train_false=defaultdict(list)
with open('data/train_data.txt') as file:
    for line in file:
        line=line.strip().split('\t')
        if line[-1]=='1':
            train_true[int(line[0])].append(int(line[1]))
            train_true[int(line[1])].append(int(line[0]))
        else:
            train_false[int(line[0])].append(int(line[1]))
            train_false[int(line[1])].append(int(line[0]))

with open('re_train_data.txt','w') as file:
    for aut in train_true:
        file.write(str(aut)+'\t'+str(train_true[aut])+'\t'+str(train_false[aut])+'\n')
