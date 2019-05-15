
import torch
import random
import numpy as np
import scipy.sparse as sp
from pandas.io.parsers import read_csv
from utils import sparse_mx_to_torch_sparse_tensor

def get_pairs(file):
    """ get gid and did as a pair from file and add it in a set"""
    df = read_csv(file)  # sep='\t'
    pairs = set()
    for i in range(0, len(df)):
        pair = (str(df['gid'][i]), df['did'][i])
        pairs.add(pair)
    return pairs


def get_items(pairs):
    """get gene_set and  disease_set from pairs set"""
    gene_set = set()
    disease_set = set()
    for pair in pairs:
        gene_set.add(pair[0])
        disease_set.add(pair[1])
    return gene_set, disease_set


def generate_negative(genes, diseases, pos_samples, nega_weight,all_pos_samples,node_index):
    """ generate negative sample by random change one of a pairs"""
    pairs = set()
    genes, diseases = list(genes), list(diseases)
    for ps in pos_samples:
        for k in range(0, nega_weight):
            index1 = random.randint(0, len(genes) - 1)
            d = ps[1]
            while True:
                if (genes[index1],d) not in all_pos_samples:
                    break
                index1 = random.randint(0, len(genes) - 1)
            pairs.add((genes[index1], d))
    return list(pairs)

def assign_index(all_genes, all_diseases):
    """ Set node index"""
    node_index = dict()
    for gene in all_genes:
        node_index[gene] = len(node_index)
    for disease in all_diseases:
        node_index[disease] = len(node_index)
    return node_index


def node_vector(item, node2index):
    n1, n2 = item[0], item[1]
    tensor1 = torch.tensor(node2index[n1])
    tensor2 = torch.tensor(node2index[n2])
    return tensor1, tensor2


def target_vector(flag):
    """ Generate label vector by flag"""
    tensor = torch.LongTensor(1).zero_()
    tensor[0] = flag
    return tensor


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def adjacency_matrix(pos_samples, ggi, dds, node2index, reweight):
    values = [1.0 for i in range(0, len(node2index))]
    vertex_1 = list(range(0, len(node2index)))
    vertex_2 = list(range(0, len(node2index)))
    """ Generate associations based on positive sample sets"""
    for ps in pos_samples:
        values.append(1.0-reweight) ## ps[2]
        vertex_1.append(node2index[ps[0]])
        vertex_2.append(node2index[ps[1]])
    """ Generate associations based on gene to gene associations"""
    df = read_csv(ggi)
    for i in range(0, len(df)):
        g1 = node2index[str(df['gid1'][i])]
        g2 = node2index[str(df['gid2'][i])]
        values.append(df['score'][i]/2.0)
        vertex_1.append(g1)
        vertex_2.append(g2)

    """ Generate associations based on disease to disease associations"""
    df = read_csv(dds)
    for i in range(0, len(df)):
        d1 = node2index[str(df['did1'][i])]
        d2 = node2index[str(df['did2'][i])]
        values.append(df['score'][i]*5.0)
        vertex_1.append(d1)
        vertex_2.append(d2)
    """"build graph, coo_matrix((data, (i, j)), [shape=(M, N)])"""
    adj = sp.coo_matrix((values, (vertex_1, vertex_2)),
                        shape=(len(node2index), len(node2index)), dtype=np.float32)
    """ build symmetric adjacency matrix"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)  # + sp.eye(adj.shape[0])
    return adj


def merge_samples(nega_weight, p_samples, n_samples):
    samples = []
    for i in range(0, min(len(p_samples), len(n_samples) // nega_weight)):
        samples.append(p_samples[i])
        for j in range(i * nega_weight, (i + 1) * nega_weight):
            samples.append(n_samples[j])
    return samples
