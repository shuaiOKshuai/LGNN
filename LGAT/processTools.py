#encoding=utf-8
import numpy as np
import pickle as pkl
import networkx as nx
import sys
import time
from numpy import dtype
import random
import scipy as spy
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def load_data_unlabeled(rootdir, dataset_str): 
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(rootdir+"ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0): 
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(rootdir+"ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil() 
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    node_num = adj.shape[0]
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_unlabeled = range(len(y)+500, node_num-len(idx_test)) 

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    unlabeled_mask = sample_mask(idx_unlabeled, labels.shape[0])
    
    
    y_train = np.zeros(labels.shape) 
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, unlabeled_mask, test_mask, idx_unlabeled


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features_new = r_mat_inv.dot(features)
    features_01 = features.todense()
    return features_new.todense(), sparse_to_tuple(features_new), features_01

def readNodesFromFile(node_file, datasetName):
    remove_first_two_values_in_features = False
    nodes_num = 0
    features_num = 0
    node_label=None 
    label_nodes={} 
    features = None
    with open(node_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)<=0:
                continue
            arr=tmp.split()
            if len(arr)>2:
                n = int(arr[0])
                l = int(arr[1])
                node_label[n]=l 
                if l in label_nodes: 
                    label_nodes[l].append(n)
                else:
                    label_nodes[l]=[n]
                if not remove_first_two_values_in_features: 
                    features[n] = arr[2:] 
                else: 
                    features[n] = arr[4:] 
                continue
            if len(arr)==2: 
                nodes_num=int(arr[0])
                if not remove_first_two_values_in_features:
                    features_num=int(arr[1])
                else: 
                    features_num=int(arr[1])-2
                node_label = np.zeros((nodes_num,), dtype=np.int32)
                features=np.zeros((nodes_num, features_num), dtype=np.float32)
    f.close()
    label_nodes_list = []
    for i in range(len(label_nodes)):
        label_nodes_list.append(label_nodes[i])
    
    if 'reddit' in datasetName: 
        features[:, 0] = np.log(features[:, 0] + 1.0)
        features[:, 1] = np.log(features[:, 1] - min(np.min(features[:, 1]), -1))
        features = features / features.sum(axis=1)[:,None]
    else: 
        features = features / features.sum(axis=1)[:,None]
    
    return node_label, label_nodes_list, features


def readEdgesFromFile_sparse(edge_file, nodes_num):
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes_num))
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                graph.add_edge(int(arr[0]), int(arr[1]))
    adj = nx.adjacency_matrix(graph)
    neis_num = adj.sum(axis=1)
    neis_num = np.squeeze(neis_num.A)
    return adj, neis_num


def preprocess_adj_normalization(adj):
    num_nodes = adj.shape[0] 
    adj = adj + np.eye(num_nodes)  
    adj[adj > 0.0] = 1.0
    D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5)) 
    adjNor = np.dot(np.dot(D_, adj), D_)
    return adj, adjNor

def preprocess_adj_normalization_sparse(adj):
    num_nodes = adj.shape[0] 
    ones = np.ones((num_nodes))
    ones_diag = sp.spdiags(ones, 0, ones.size, ones.size) 
    adj_self = adj + ones_diag  
    tmp = np.sum(adj_self, axis=1)
    tmp = tmp.A
    tmp = np.squeeze(tmp)
    tmp = np.power(tmp, -0.5)
    D_ = sp.spdiags(tmp, 0, tmp.size, tmp.size) 
    adj_self_nor = np.dot(np.dot(D_, adj_self), D_)
    return adj_self, adj_self_nor


def processNodeInfo_sparse_dense_maxDegree(adj, mask_nor, nodes_num, max_degree):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis_row = []
    neis_col = []
    neis_value = []
    neis_mask_row = []
    neis_mask_col = []
    neis_mask_value = []
    neis_mask_nor_row = []
    neis_mask_nor_col = []
    neis_mask_nor_value = []
    AB = adj.nonzero()
    A = AB[0] 
    B = AB[1] 
    former = -1 
    col_index = 0 
    for i in range(len(A)):
        if A[i] != former: 
            col_index = 0
            former = A[i]
        neis_row.append(A[i])
        neis_mask_row.append(A[i])
        neis_mask_nor_row.append(A[i])
        neis_col.append(col_index)
        neis_mask_col.append(col_index)
        neis_mask_nor_col.append(col_index)
        neis_value.append(B[i])
        neis_mask_value.append(1.0)
        neis_mask_nor_value.append(mask_nor[A[i], B[i]])
        col_index += 1
    neis = sp.csc_matrix((spy.array(neis_value),(spy.array(neis_row), spy.array(neis_col))),shape=(nodes_num, max_nei_num))
    neis = neis.toarray() 
    
    neis_new = np.zeros((nodes_num, max_degree), dtype=np.int32)
    degrees = np.squeeze(np.sum(adj, axis=1).A.astype(np.int32))
    for i in range(nodes_num): 
        neighbour = None
        if degrees[i] >= max_degree: 
            neighbour = np.random.choice(neis[i, :degrees[i]], max_degree, replace=False) 
        else: 
            neighbour1 = neis[i, :degrees[i]] 
            neighbour2 = np.random.choice(neis[i, :degrees[i]], max_degree-degrees[i], replace=True) 
            neighbour = np.concatenate((neighbour1, neighbour2))
            np.random.shuffle(neighbour)
            
        neis_new[i] = neighbour
    return neis_new


def prepareDataset(nodes_num, max_degree, node_ids, neisMatrix, neisMatrix_mask, neisMatrix_mask_nor):
    nodes_self = np.arange(nodes_num, dtype=np.int32)
    ones_nodes_self = np.ones(nodes_num, dtype=np.float32)
    neisMatrix = np.concatenate((nodes_self[:,None], neisMatrix), axis=1) 
    neisMatrix_mask = np.concatenate((ones_nodes_self[:,None], neisMatrix_mask), axis=1) 
    neisMatrix_mask_nor = np.concatenate((ones_nodes_self[:,None], neisMatrix_mask_nor), axis=1) 
    max_degree = max_degree + 1
    
    adj_1 = neisMatrix[node_ids] 
    mask_1 = neisMatrix_mask[node_ids] 
    mask_1_nor = neisMatrix_mask_nor[node_ids] 
    
    adj_1_max_nei_num = np.max(np.sum(mask_1, axis=1)).astype(np.int32)
    adj_1 = adj_1[:,:adj_1_max_nei_num]
    mask_1 = mask_1[:,:adj_1_max_nei_num]
    mask_1_nor = mask_1_nor[:,:adj_1_max_nei_num]
    
    valid_node_num_adj_1 = np.sum(mask_1).astype(np.int32)
    adj_1_new = np.zeros_like(adj_1).astype(np.int32) 
    adj_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.int32)
    mask_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    mask_2_nor = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    index = 0 
    for i in range(adj_1.shape[0]):
        nodeId = node_ids[i]
        valid_num_row = np.sum(mask_1[i]).astype(np.int32)
        valid_ids_row = adj_1[i, :valid_num_row]
        adj_2[index : index+valid_num_row] = neisMatrix[valid_ids_row] 
        mask_2[index:index+valid_num_row] = neisMatrix_mask[valid_ids_row]
        mask_2_nor[index:index+valid_num_row] = neisMatrix_mask_nor[valid_ids_row]
        adj_1_new[i, :valid_num_row] = np.arange(index, index+valid_num_row)
        
        index += valid_num_row
    
    adj_2_max_nei_num = np.max(np.sum(mask_2, axis=1)).astype(np.int32)
    adj_2 = adj_2[:,:adj_2_max_nei_num]
    mask_2 = mask_2[:,:adj_2_max_nei_num]
    mask_2_nor = mask_2_nor[:,:adj_2_max_nei_num]
    
    return adj_2, mask_2, mask_2_nor, adj_1_new, mask_1, mask_1_nor


def prepareDataset_bigdata(nodes_num, max_degree, node_ids, neisMatrix, neisMatrix_mask, neisMatrix_mask_nor, max_degree_setting=40):
    nodes_self = np.arange(nodes_num, dtype=np.int32)
    ones_nodes_self = np.ones(nodes_num, dtype=np.float32)
    neisMatrix = np.concatenate((nodes_self[:,None], neisMatrix), axis=1) 
    neisMatrix_mask = np.concatenate((ones_nodes_self[:,None], neisMatrix_mask), axis=1) 
    neisMatrix_mask_nor = np.concatenate((ones_nodes_self[:,None], neisMatrix_mask_nor), axis=1) 
    max_degree = max_degree + 1
    max_degree_setting = max_degree_setting +1
    if max_degree > max_degree_setting: 
        max_degree = max_degree_setting
    
    adj_1 = neisMatrix[node_ids] 
    mask_1 = neisMatrix_mask[node_ids] 
    mask_1_nor = neisMatrix_mask_nor[node_ids] 
    
    adj_1_max_nei_num = np.max(np.sum(mask_1, axis=1)).astype(np.int32)
    if adj_1_max_nei_num > max_degree_setting: 
        adj_1_max_nei_num = max_degree_setting
    adj_1 = adj_1[:,:adj_1_max_nei_num]
    mask_1 = mask_1[:,:adj_1_max_nei_num]
    mask_1_nor = mask_1_nor[:,:adj_1_max_nei_num]
    mask_1_nor[:, 1:] = mask_1[:, 1:] / np.sum(mask_1[:, 1:], axis=-1)[:,None]
    mask_1_nor[np.isinf(mask_1_nor)] = 0. 
    
    valid_node_num_adj_1 = np.sum(mask_1).astype(np.int32)
    adj_1_new = np.zeros_like(adj_1).astype(np.int32) 
    adj_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.int32)
    mask_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    mask_2_nor = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    index = 0 
    for i in range(adj_1.shape[0]):
        nodeId = node_ids[i]
        valid_num_row = np.sum(mask_1[i]).astype(np.int32)
        valid_ids_row = adj_1[i, :valid_num_row]
        adj_2[index : index+valid_num_row] = neisMatrix[valid_ids_row, :max_degree] 
        mask_2[index:index+valid_num_row] = neisMatrix_mask[valid_ids_row, :max_degree]
        mask_2_nor[index:index+valid_num_row] = neisMatrix_mask_nor[valid_ids_row, :max_degree]
        adj_1_new[i, :valid_num_row] = np.arange(index, index+valid_num_row)
        
        index += valid_num_row
    
    adj_2_max_nei_num = np.max(np.sum(mask_2, axis=1)).astype(np.int32)
    adj_2 = adj_2[:,:adj_2_max_nei_num]
    mask_2 = mask_2[:,:adj_2_max_nei_num]
    mask_2_nor = mask_2_nor[:,:adj_2_max_nei_num]
    mask_2_nor[:, 1:] = mask_2[:, 1:] / np.sum(mask_2[:, 1:], axis=-1)[:,None]
    mask_2_nor[np.isinf(mask_2_nor)] = 0. 
    
    return adj_2, mask_2, mask_2_nor, adj_1_new, mask_1, mask_1_nor


def processNodeInfo(adj, mask_nor, node_num):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis = np.zeros((node_num,max_nei_num), dtype=np.int32)
    neis_mask = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neis_mask_nor = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neighboursDict = []
    inner_index = 0 
    for i in range(node_num):
        inner_index = 0
        nd = [] 
        for j in range(node_num):
            if adj[i][j]==1.0: 
                neis[i][inner_index] = j 
                neis_mask[i][inner_index] = 1.0
                neis_mask_nor[i][inner_index] = mask_nor[i][j]
                if i!=j: 
                    nd.append(j)
                inner_index += 1
        neighboursDict.append(nd)
    
    return neis, neis_mask, neis_mask_nor, neighboursDict

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def micro_macro_f1_removeMiLabels(y_true, y_pred, labelsList):
    return f1_score(y_true, y_pred, labels = labelsList, average="micro"), f1_score(y_true, y_pred, average="macro")


def generateTrainValTs(nodeFile, trainNum_perClass=20, valAllNum=500, testAllNum=1000):
    np.random.seed(123)
    random.seed(123)
    labels_nodes={}
    node_num=0
    feature_num=0
    with open(nodeFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                if len(arr)==2:
                    node_num=int(arr[0])
                    feature_num=int(arr[1])
                    continue
                id=int(arr[0])
                label=int(arr[1])
                if label in labels_nodes:
                    labels_nodes[label].append(id)
                else:
                    ls=[id]
                    labels_nodes[label]=ls
    train=[]
    for label in labels_nodes:
        train.extend(random.sample(labels_nodes[label], trainNum_perClass))
    train_set = set(train)
    all_set=set(range(node_num))
    left_set = all_set - train_set
    left_list=list(left_set)
    random.shuffle(left_list)
    test=left_list[-testAllNum:] 
    val=left_list[-(valAllNum+testAllNum):-testAllNum] 
    return train, val, test


def load_data_other_dataset(dataset, train_pre, val_pre, test_pre, nodeFile, edgeFile):
    node_num = 0
    feature_num = 0
    features = None
    index = 0
    nodeLabelsArray=None
    labelsSet = set()
    lineCount = 0
    with open(nodeFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                if len(arr)==2 and lineCount==0:
                    print(arr)
                    node_num=int(arr[0])
                    feature_num=int(arr[1])
                    print('node num == ', node_num)
                    print('feature num == ', feature_num)
                    features=np.zeros((node_num, feature_num))
                    nodeLabelsArray = np.zeros((node_num,), dtype=np.int32)
                    lineCount += 1
                    continue
                node_id = int(arr[0])
                features[node_id]=arr[2:] 
                nodeLabelsArray[node_id] = int(arr[1]) 
                labelsSet.add(int(arr[1]))
                index+=1
                lineCount += 1
    label_num = len(labelsSet)
    adj=np.zeros((node_num,node_num))
    with open(edgeFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                adj[int(arr[0]), int(arr[1])]=1.0
    y_train = np.zeros((node_num,label_num))
    y_val = np.zeros((node_num,label_num))
    y_test = np.zeros((node_num,label_num))
    train_mask = np.zeros((node_num,))
    val_mask = np.zeros((node_num,))
    test_mask = np.zeros((node_num,))
    nodeLabels = list(nodeLabelsArray)
    
    for id in train_pre:
        y_train[id, nodeLabels[id]] = 1.0
        train_mask[id] = 1.0
    for id in val_pre:
        y_val[id, nodeLabels[id]] = 1.0
        val_mask[id] = 1.0
    for id in test_pre:
        y_test[id, nodeLabels[id]] = 1.0
        test_mask[id] = 1.0
    
    features_01 = features
    if dataset in {"amazon"}:
        fea_norm = np.linalg.norm(features, axis=1)
        features_new = features / fea_norm[:,None]
    else: 
        features_new = preprocess_features_other_dataset(features)
    
    features = np.mat(features_new)
    features_01 = np.mat(features_01)
    
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    test_mask = np.array(test_mask, dtype=np.bool)
    
    return adj, features, features_01, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features_other_dataset(features):
    """Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features_new = r_mat_inv.dot(features)
    return features_new