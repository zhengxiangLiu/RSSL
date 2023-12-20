import numpy as np
from scipy.sparse import csr_matrix
from collections import deque
import torch
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
from torch_sparse import coalesce
from operator import itemgetter
from sklearn.decomposition import TruncatedSVD
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i])
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

class get_trees():
    def __init__(self,tree2s):
        matrix_size = len(tree2s)
        weight_matrix = np.zeros((matrix_size, matrix_size))

        for tree in tree2s:
            root_index = list(tree.keys())[0]
            # root_index = ticker_index[root_node]
            # weight_matrix[root_index, root_index] = 0.5

            queue = deque([(root_index, 1)])
            while queue:
                node, level = queue.popleft()
                children = tree[node]
                for child_index in children:
                    # child_index = ticker_index[child]
                    weight = 0.6 / level
                    weight_matrix[root_index, child_index] = weight
                    weight_matrix[child_index, root_index] = weight
                    queue.append((child_index, level + 2))
        # weight_matrix = np.where(weight_matrix < 0.1, 0.0, weight_matrix)

        self.tree_data=weight_matrix


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)
class DataGraph():
    def __init__(self,data,lengh):
        graphdata = np.asarray(data[0], dtype=object)
        data_index=np.asarray(data[1], dtype=object)
        graph=np.zeros((lengh, lengh))
        for raw,index in zip(graphdata,data_index):
             for i in raw:
                graph[index][i] = 1


        graph_sparse = sp.coo_matrix(graph)
        edge_index, _ = from_scipy_sparse_matrix(graph_sparse)
        # edge_index = edge_index.cuda()

        # device = torch.device("cuda")
        # indices = torch.LongTensor([graph_sparse.row, graph_sparse.col]).to(device)
        # values = torch.FloatTensor(graph_sparse.data).to(device)
        # shape = torch.Size(graph_sparse.shape)
        # graph_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)
        self.stock_pca=graph


class get_kalma():
    def __init__(self,stock_data):
        transition_matrix=[]
        transition_covariance=[]
        for n in range(len(stock_data)):
            data=stock_data[-100:, n:n + 4, :]
            transition_matrix_item = np.zeros((9, 9))
            transition_covariance_item = np.zeros((9, 9))
            for i in range(3):
                for j in range(9):
                    for k in range(9):
                        transition_matrix_item[j][k] += np.corrcoef(data[:, i, j], data[:, i + 1, k])[0, 1]
                        xyz = data[:, i, j] - data[:, i + 1, k]
                        transition_covariance_item[j][k] += np.var(xyz)
            transition_matrix_item /= 3
            transition_covariance_item /= 3
            transition_matrix.append(transition_matrix_item)
            transition_covariance.append(transition_covariance_item)
        self.transition_matrix=transition_matrix
        self.transition_covariance=transition_covariance
        initial_state_mean=[]
        initial_cov=[]
        for n in range(len(stock_data)):

            data=stock_data[-100:, n:n + 4, :]
            stds = np.std(data, axis=(0, 1))
            initial_cov.append(np.diag(stds ** 2))
            initial_state=0
            for i in range(4):
                initial_state += np.mean(data[:,i,:], axis=0)
            initial_state_mean.append(initial_state/4)
        self.initial_state_mean=initial_state_mean
        self.initial_cov=initial_cov







class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0],dtype=object)
        H_T = data_masks(self.raw, n_node)

        b_test=1.0/H_T.sum(axis=1).reshape(1, -1)
        b_test2=H_T.sum(axis=1).reshape(1, -1)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        k=H.sum(axis=1)
        data_weight1 = np.asarray(H.sum(axis=1).reshape(1, -1))[0]
        data_weight2 = np.asarray(H.sum(axis=1).reshape(1, -1))
        for i in range(len(data_weight1)):
            if (data_weight1[i] == 0):
                data_weight2[0][i] = 1
        DH = H.T.multiply(1.0/data_weight2)
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)

        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask
