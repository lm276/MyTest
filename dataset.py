import torch
from dgl.data import CoraGraphDataset, CitationGraphDataset
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import networkx as nx
import numpy as np
from numpy import random,mat
import os
import torch as th

def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer' or 'pubmed':
        return CitationGraphDataset(name=dataset)
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)
        graph=ds[0]
        nxg = graph.to_networkx()
        adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

        adj = preprocess_adj(adj)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        # adj = graph.adjacency_matrix(transpose=True)
        diff = compute_ppr(graph, 0.2)
        feat = graph.ndata['feat']
        labels = graph.ndata['label']

        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        idx_train = th.nonzero(train_mask, as_tuple=False).squeeze()
        idx_val = th.nonzero(val_mask, as_tuple=False).squeeze()
        idx_test = th.nonzero(test_mask, as_tuple=False).squeeze()
        # idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
        # idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
        # idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)
        
        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
        adj = adj.todense()
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')
        #adj=mat(adj)
        adj=sp.coo_matrix(adj.any(), dtype=sp.coo_matrix)
        adj=adj.astype(float)
        adj = adj.todense()

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)



    return adj, diff, feat, labels, idx_train, idx_val, idx_test
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    load('cora')