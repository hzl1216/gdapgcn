import torch
import gensim
import _pickle as cPickle
import numpy as np
import scipy.sparse as sp
from dataset_tool import normalize
from model import GCN
import torch.nn.functional as F
from file_name import files_name
import os
from datetime import datetime
from utils import sparse_mx_to_torch_sparse_tensor
def load_pretrain_vector(n2i_f,vec_dir):
    global node2index
    node2index = cPickle.load(n2i_f)
    my_word2vec = gensim.models.Word2Vec.load(vec_dir)
    embedding = [ [] for i in range(0,len(node2index.keys() )) ]
    for key in node2index.keys() :
        index = node2index[key]
        embedding[index] = my_word2vec.wv[key].tolist()
    embedding = np.array(embedding)
    print (len(node2index))
    print (embedding.shape)
    return torch.from_numpy(embedding)


def get_avg_rp(rp):
    '''
    calculate the cosine_similarity between all node
    :param rp:feature matrix
    :return:
    '''
    rp_data = rp.data
    threshold = 0.4
    cos_sm = []
    for i in range(0,rp_data.size()[0]):
        if i % 1000 == 0 :
            print (i)
        rp1 = torch.unsqueeze(rp_data[i],0)
        column = F.cosine_similarity(rp1,rp_data)
        cos_sm.append(column)
    cos_sm = torch.stack(cos_sm,0)
    threshold = torch.DoubleTensor([[threshold]])
    bit_sm = cos_sm > threshold # Fullfil with 0 or 1
    
    print (torch.sum(bit_sm.data))
    return cos_sm, bit_sm


def get_block_count(bit_sm):
    '''
    count the g-g,g-d,d-d associations the cosine_similarity above threshold
    :param bit_sm:
    :return:
    '''
    global node2index,gene_range,disease_range
    gene_list, disease_list = [],[]
    
    for key in node2index.keys():
        index = node2index[key]
        if key[0] == 'C' :
            disease_list.append(index)
        else:
            gene_list.append(index)
            
    gene_range = (min(gene_list),max(gene_list))
    disease_range = (min(disease_list),max(disease_list))
    print( gene_range, disease_range)
    gg_edges = torch.sum(bit_sm[:gene_range[1],:gene_range[1]])/2
    dd_edges = torch.sum(bit_sm[disease_range[0]:,disease_range[0]:])/2
    gd_edges = torch.sum(bit_sm[:gene_range[1],disease_range[0]:])
    
    print (gg_edges, dd_edges, gd_edges)


def save_sparse_cossm(cos_sm, bit_sm, target_file):
    '''
    Save similarity sparse matrix
    :param cos_sm: cos similarity matrix
    :param bit_sm:bit similarity matrix
    :param target_file:save dir
    :return:
    '''
    cos_sparse = torch.mul(cos_sm, bit_sm.double())
    values, vertex_1, vertex_2 = [],[],[]
    for i in range(0,cos_sparse.size()[1]):
        row_sp = cos_sparse[i].to_sparse()
        index2 = row_sp.indices()[0]
        index1 = [ i for j in range(0,index2.size()[0]) ]
        value = row_sp.values()
        values.extend(value.tolist())
        vertex_1.extend(index1)
        vertex_2.extend(index2.tolist())
    adj = sp.coo_matrix((values, (vertex_1, vertex_2)),
                         shape=tuple(cos_sparse.size()), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = normalize(adj) 
    #cos_sparse = cos_sparse.to_sparse()
    f = open(target_file, 'wb')
    cPickle.dump(adj, f)
    return values, vertex_1, vertex_2



def load_trained_vector(epoch,number,n2i_f,file_homes):
    global node2index
    node2index = cPickle.load(n2i_f)
    node_count = len(node2index)
    node_dim = 128
    n_repr = 128
    gcn = GCN(node_count,node_dim,n_repr)
    gcn.load_state_dict(torch.load(file_homes+'/networks/GCN_%d_%d.pth'%(number,epoch),map_location='cpu'))
    f = open(files_home + '/networks/adj_matrix_%d_full' % (number), 'rb')
    full_adj_matrix = cPickle.load(f)
    full_adj_matrix = sparse_mx_to_torch_sparse_tensor(full_adj_matrix)
    init_input = torch.LongTensor([j for j in range(0, node_count)])
    gcn.eval()

    rp_matrix = gcn(init_input, full_adj_matrix)
    #gcn.to(device)
    return rp_matrix.double()

def main(files_home):
    starttime = datetime.now()
    print('start calculate similarity ',starttime)
    n2i_f = open(os.path.join(files_home, files_name['node_index']), 'rb')
    vec_dir = os.path.join(files_home, files_name['word2vec'])
    target_file = os.path.join(files_home, files_name['similarity_matrix'])

    rp_matrix = load_pretrain_vector(n2i_f,vec_dir)
#    rp_matrix = load_trained_vector(99,number,n2i_f,files_home)
    cos_sm, bit_sm = get_avg_rp(rp_matrix)
    get_block_count(bit_sm)
    save_sparse_cossm(cos_sm, bit_sm, target_file)
    
    endtime = datetime.now()
    print('finish similarity calculated! run spend ',endtime-starttime)
if __name__ == '__main__':
    files_home = files_name['file_home']
    main(files_home,2)
