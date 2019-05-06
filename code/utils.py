import numpy as np
import torch
from gensim.models import Word2Vec
def read_file(fname):

    f = open(fname,'r')
    line = f.readline()
    series = []

    while line :
        # must be aware here, [:-1] or [:-2] in different situation
        # make sure the vocab size is equal to the total nodes
        line = line[:-1] # '\r\n'
        serie = line.split('\t')
        series.append(serie)
        line = f.readline()
    f.close()

    return series

def skip_gram(infile, outfile):

    series = read_file(infile)

    model = Word2Vec(series, sg=1, size=128, min_count=5, iter=5, window=10)
    print(len(model.wv.vocab))
    model.save(outfile)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_pretrain_vector(node2index,vec_dir):
    my_word2vec = Word2Vec.load(vec_dir)
    embedding = [ [] for i in range(0,len(node2index.keys() )) ]
    for key in node2index.keys() :
        index = node2index[key]
        embedding[index] = my_word2vec.wv[key].tolist()
    embedding = np.array(embedding)
    return embedding

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
              np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
