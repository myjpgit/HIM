
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix,lil_matrix,eye
import csv
from collections import defaultdict
import scipy

def author_index():
    all_authors = set()
    with open('refined_data/refined_true_aa.csv', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            all_authors.add(line[0])
            all_authors.add(line[1])
    with open('refined_data/cut_false_aa_by_diff_year.csv', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            all_authors.add(line[0])
            all_authors.add(line[1])
    print(len(all_authors))
    with open('refined_data/author_index.txt','w',encoding='utf-8') as file:
        for aut in all_authors:
            file.write(aut+'\n')

def get_author_index():
    aut_ind={}
    count=0
    with open('refined_data/author_index.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            aut_ind[line]=count
            count+=1
    return aut_ind

def institution_index():
    all_insts=set()
    with open('refined_data/author_institution.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            inss=eval(con[1])
            for i in inss:
                i=i.strip()
                all_insts.add(i)
    # print(len(all_insts))
    with open('refined_data/institution_index.txt','w',encoding='utf-8') as file:
        for ins in all_insts:
            file.write(ins+'\n')


def get_institution_index():
    ins_ind={}
    count=0
    with open('refined_data/institution_index.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            ins_ind[line]=count
            count+=1
    return ins_ind

def author_institution_matrix():
    aut_ind=get_author_index()
    aut_num=len(aut_ind)
    ins_ind=get_institution_index()
    ins_num=len(ins_ind)
    aut_ins_ma=lil_matrix((aut_num,ins_num),dtype=np.float32)
    with open('refined_data/author_institution.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            aut_id=aut_ind[con[0]]
            inss=eval(con[1])
            for i in inss:
                i=i.strip()
                ins_id=ins_ind[i]
                aut_ins_ma[aut_id,ins_id]=1.0
    aut_ins_ma=csr_matrix(aut_ins_ma)
    aim={}
    aim['author_institution']=aut_ins_ma
    sio.savemat('refined_data/aut_institution',aim)

def paper_attribute_index():
    att_num=defaultdict(int)
    with open('refined_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            info=eval(con[1])
            att_num[str(info['year'])]+=1
    with open('refined_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            info=eval(con[1])
            if info.__contains__('fos'):
                for f in info['fos']:
                    att_num[f]+=1
            if info.__contains__('keywords'):
                for k in info['keywords']:
                    att_num[k]+=1
    with open('refined_data/paper_attribute_index.txt','w',encoding='utf-8') as file:
        for att,num in att_num.items():
            if num>1:
                file.write(str(att)+'\n')

def get_paper_attribute_index():
    att_ind={}
    count=0
    with open('refined_data/paper_attribute_index.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            att_ind[line]=count
            count+=1
    return att_ind

def get_paper_index():
    paper_ind={}
    count=0
    with open('refined_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip().split('\t')
            paper_ind[line[0]]=count
            count+=1
    return paper_ind

def paper_attribute_matrix():
    att_ind=get_paper_attribute_index()
    paper_ind=get_paper_index()
    paper_att_ma=lil_matrix((len(paper_ind),len(att_ind)),dtype=np.float32)
    # print(paper_att_ma.shape)
    with open('refined_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip().split('\t')
            con=eval(line[1])
            if con.__contains__('keywords'):
                for k in con['keywords']:
                    if att_ind.__contains__(k):
                        paper_att_ma[paper_ind[line[0]],att_ind[k]]+=1.0
            if con.__contains__('fos'):
                for f in con['fos']:
                    if att_ind.__contains__(f):
                        paper_att_ma[paper_ind[line[0]],att_ind[f]]+=1.0
    # print(paper_att_ma)
    pam={}
    pam['paper_attribute']=csr_matrix(paper_att_ma)
    sio.savemat('refined_data/paper_attribute',pam)

def author_paper_matrix():
    aut_ind=get_author_index()
    paper_ind=get_paper_index()
    aut_pap_ma=lil_matrix((len(aut_ind),len(paper_ind)),dtype=np.float32)
    with open('refined_data/author_paper.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip().split('\t')
            ps=eval(line[1])
            for p in ps:
                if paper_ind.__contains__(p):
                    aut_pap_ma[aut_ind[line[0]],paper_ind[p]]+=1.0
    apm={}
    apm['author_paper']=csr_matrix(aut_pap_ma)
    sio.savemat('refined_data/author_paper',apm)

def author_author_matrix():
    aut_ind=get_author_index()
    aut_aut_ma=lil_matrix((len(aut_ind),len(aut_ind)),dtype=np.float32)
    with open('refined_data/refined_true_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            ind1=aut_ind[line[0]]
            ind2=aut_ind[line[1]]
            aut_aut_ma[ind1,ind2]+=1.0
            aut_aut_ma[ind2,ind1]+=1.0
    with open('refined_data/cut_false_aa_by_diff_year.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            ind1=aut_ind[line[0]]
            ind2=aut_ind[line[1]]
            aut_aut_ma[ind1,ind2]+=1.0
            aut_aut_ma[ind2,ind1]+=1.0
    aam={}
    aam['author_author']=csr_matrix(aut_aut_ma)
    sio.savemat('refined_data/author_author',aam)

def one_hot_vector():
    aut_ind=get_author_index()
    ins_ind=get_institution_index()
    aut_num=len(aut_ind)
    ind_num=len(ins_ind)
    aut_att=eye(aut_num,dtype=np.float32)
    ins_att=eye(ind_num,dtype=np.float32)
    oh={}
    oh['author_attribute']=csr_matrix(aut_att)
    oh['institution_attribute']=csr_matrix(ins_att)
    sio.savemat('refined_data/one_hot_vector',oh)

def get_year_index():
    count=0
    year_ind={}
    with open('refined_data/paper_attribute_index.txt') as file:
        for line in file:
            line=line.strip()
            year_ind[line]=count
            count+=1
            if count>=48:
                break
    return year_ind

def re_paper_attribute_matrix():
    year_ind=get_year_index()
    # print(len(year_ind))
    year_num=len(year_ind)
    paper_ind=get_paper_index()
    paper_year_ma=lil_matrix((len(paper_ind),len(year_ind)+len(paper_ind)),dtype=np.float32)

    with open('refined_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            info=eval(con[1])
            year=int(info['year'])
            if year in year_ind:
                paper_year_ma[paper_ind[line[0]],year_ind[year]]=1.0
    for i in range(len(paper_ind)):
        paper_year_ma[i,i+year_num]=1.0
    pam={}
    pam['paper_attribute']=csr_matrix(paper_year_ma)
    sio.savemat('refined_data/paper_attribute',pam)



if __name__=='__main__':
    # author_index()
    # institution_index()
    # author_institution_matrix()
    # paper_attribute_index()
    # paper_attribute_matrix()
    # author_paper_matrix()
    # author_author_matrix()
    one_hot_vector()
    re_paper_attribute_matrix()

