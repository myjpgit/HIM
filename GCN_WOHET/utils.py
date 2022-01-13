
import scipy.sparse as sp
import scipy.io as sio
import numpy as np

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def preprocess_lap_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj_normalized = adj - sp.eye(adj.shape[0]) #拉普拉斯矩阵设置参数
    return adj_normalized

def extract_feature(data):
    one_hot = sio.loadmat('../' + data + '/refined_data/one_hot_vector.mat')
    if data=='advisor_advisee':
        # aut_att=one_hot['author_attribute']
        ins_att=one_hot['institution_attribute']
        paper_att=sio.loadmat('../'+data+'/refined_data/paper_attribute.mat')
        pap_att=paper_att['paper_attribute']
        return ins_att,pap_att
    elif data=='enron':
        ema_att=one_hot['email_attribute']
        # emp_att=one_hot['employee_attribute']
        return ema_att
    elif data=='epinion':
        item_att_1=one_hot['item_1']
        item_att_2=one_hot['item_2']
        return item_att_1,item_att_2
    elif data=='terror_attack':
        fea_att=one_hot['feature']
        return fea_att
    elif data=='same_city':
        fea_att=one_hot['feature']
        return fea_att


def extract_adj(data):

    if data=='advisor_advisee':
        author_author=sio.loadmat('../'+data+'/refined_data/author_author.mat')
        aut_aut_1=author_author['author_author_1']
        aut_aut_2=author_author['author_author_2']
        author_institution=sio.loadmat('../'+data+'/refined_data/aut_institution.mat')
        aut_ins=author_institution['author_institution']
        author_paper=sio.loadmat('../'+data+'/refined_data/author_paper.mat')
        aut_pap=author_paper['author_paper']
        adj=np.load('../'+data+'/refined_data/adj.npy')
        efeature=np.load('../'+data+'/refined_data/efeature.npy')
        return aut_aut_1,aut_aut_2,aut_ins,aut_pap,np.array(adj),np.array(efeature)
    elif data=='enron':
        employee_employee=sio.loadmat('../' + data + '/refined_data/employee_employee.mat')
        emp_emp_1=employee_employee['employee_employee_1']
        emp_emp_2 = employee_employee['employee_employee_2']
        employee_email=sio.loadmat('../' + data + '/refined_data/employee_email.mat')
        emp_ema=employee_email['employee_email']
        return emp_emp_1,emp_emp_2,emp_ema
    elif data=='epinion':
        user_item=sio.loadmat('../'+data+'/refined_data/user_item.mat')
        user_user=sio.loadmat('../'+data+'/refined_data/user_user.mat')
        adj=np.load('../'+data+'/refined_data/adj.npy')
        efeature=np.load('../'+data+'/refined_data/efeature.npy')
        return user_user['user_user_1'],user_user['user_user_2'],user_item['user_item_1'],user_item['user_item_2'],np.array(adj),np.array(efeature)
    elif data=='terror_attack':
        node_node=sio.loadmat('../'+data+'/refined_data/node_node.mat')
        node_feature = sio.loadmat('../' + data + '/refined_data/node_feature.mat')
        adj=np.load('../'+data+'/refined_data/adj.npy')
        efeature=np.load('../'+data+'/refined_data/efeature.npy')
        return node_node['node_node_1'],node_node['node_node_2'],node_feature['node_feature'],np.array(adj),np.array(efeature)
    elif data=='same_city':
        node_node=sio.loadmat('../'+data+'/refined_data/node_node.mat')
        node_feature = sio.loadmat('../' + data + '/refined_data/node_feature.mat')
        adj=np.load('../'+data+'/refined_data/adj.npy')
        efeature=np.load('../'+data+'/refined_data/efeature.npy')
        return node_node['node_node_1'], node_node['node_node_2'], node_feature['node_feature'], np.array(adj), np.array(efeature)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    #r_inv = np.power(rowsum, -1).flatten()
    r_inv = np.power(rowsum + 1e-6, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def extract_val_data(data):
    row_edge_idx=[]
    col_pos_edge_idx=[]
    with open('../{}/refined_data/valid_data.txt'.format(data)) as file:
        for line in file:
            line=line.strip().split('\t')
            row_edge_idx.append(int(line[1]))
            col_pos_edge_idx.append(int(line[0]))
    return row_edge_idx,col_pos_edge_idx

def extract_test_data(data):
    row_edge_idx=[]
    col_pos_edge_idx=[]
    with open('../{}/refined_data/test_data.txt'.format(data)) as file:
        for line in file:
            line=line.strip().split('\t')
            row_edge_idx.append(int(line[1]))
            col_pos_edge_idx.append(int(line[0]))
    return row_edge_idx,col_pos_edge_idx

def write_result(data,res):
    with open('../{}/refined_data/result.txt'.format(data),'w') as file:
        for r in res:
            file.write(str(r)+'\n')

def average_rank(train_num, all_num, pre_label):
    sort_label=list(np.argsort(pre_label))
    sort_label=sort_label[::-1]
    size=len(pre_label)
    size_label = train_num * all_num
    size_all = size * size_label
    pos=0
    for i in range(len(sort_label)):
        if sort_label[i]<size:
            pos+=i
    return pos*10/size_all

def backto_categorical(X):
    for i in range(0, len(X)):
        if X[i] >= 0.5:
            X[i] = 1
        else:
            X[i] = 0