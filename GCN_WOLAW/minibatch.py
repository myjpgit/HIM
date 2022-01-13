
from collections import defaultdict
import numpy as np


class MiniBatch():
    def __init__(self,dataset,batch_size):
        self.dataset=dataset
        self.batch_size=batch_size

        self._extract_data()
        self._generate_distribution()

    def _extract_data(self):
        self.true_data=defaultdict(list)
        self.false_data=defaultdict(list)

        with open('../{}/refined_data/train_data.txt'.format(self.dataset)) as file:
            for line in file:
                line=line.strip().split('\t')
                self.true_data[int(line[0])]=eval(line[1])
                self.false_data[int(line[0])]=eval(line[2])

    def _generate_distribution(self):
        self.keys=[]
        self.distribution=[]
        for k in self.true_data:
            if len(self.true_data[k])>0 and len(self.false_data[k])>0:
                self.keys.append(k)
                self.distribution.append(len(self.true_data[k]))
        dis_sum=sum(self.distribution)
        self.distribution=[i/dis_sum for i in self.distribution]

    def _sample_true(self,key):
        return np.random.choice(self.true_data[key])
    def _sample_false(self,key):
        return np.random.choice(self.false_data[key])

    def get_batch_data(self):
        col_pos_edge_idx=[]
        col_neg_edge_idx=[]
        row_edge_idx=np.random.choice(self.keys,size=self.batch_size,p=self.distribution)
        for r in row_edge_idx:
            col_pos_edge_idx.append(self._sample_true(r))
            col_neg_edge_idx.append(self._sample_false(r))
        return row_edge_idx,np.array(col_pos_edge_idx),np.array(col_neg_edge_idx)










