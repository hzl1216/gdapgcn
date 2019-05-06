import _pickle as cPickle
from dataset import *
from model import *
from utils import load_pretrain_vector,sparse_mx_to_torch_sparse_tensor
from cluster_loss import Cluster_Loss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import argparse
from file_name import files_name
import os
import ast
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='use CUDA (default: True),input should be either "True" or "False".',
                            type=ast.literal_eval, dest='cuda')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='args.number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--load_node_index', type=ast.literal_eval, default=True, dest='load_node_index',
                            help='load node index,(default: True, False meaning renew node index),'
                                                     'input should be either "True" or "False".')
parser.add_argument('--number', type=int, default=1,
                            help='number to use sign the different train model')
args = parser.parse_args()

def main(files_home):
    starttime = datetime.now()
    print('start train model ',starttime)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda==True:
        torch.cuda.manual_seed(args.seed)
    # Load data

    f_train = os.path.join(files_home, files_name['train_file'])
    node_index_file = os.path.join(files_home, files_name['node_index'])
    ### load mode
    if args.load_node_index==True:
        f = open(node_index_file, 'rb')
        node2index = cPickle.load(f)
        trainset = Sample_Set(f_train, node2index)
    ### load mode end
    else:
        ### new mode
        trainset = Sample_Set(f_train)
        node2index = trainset.get_node2index()
        f = open(node_index_file,'wb')
        cPickle.dump(node2index,f)

    node2index_r = dict()
    for key in node2index.keys():
        value = node2index[key]
        node2index_r[value] = key

    f_ggi = os.path.join(files_home, files_name['graph_ggn_file'])
    f_dds = os.path.join(files_home, files_name['graph_dds_file'])

    node_count = len(node2index)
    node_dim = 128
    n_repr = 128

    gcn = GCN(node_count,node_dim,n_repr)
    lp_model = Link_Prediction(n_repr)

    if args.cuda == True:
        gcn.cuda()
        lp_model = nn.DataParallel(lp_model)
        lp_model.cuda()

        print('using gpu #', torch.cuda.current_device())

    vec_dir = os.path.join(files_home, files_name['word2vec'])
    embedding = load_pretrain_vector(node2index, vec_dir)  #
    gcn.embedding.weight.data.copy_(torch.from_numpy(embedding))  #
    trainset.reassign_samples(gcn.embedding.weight)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    print('Total %d samples.' % (len(trainset)))
    print('Genes: %d, Diseases: %d ' % (len(trainset.all_genes), len(trainset.all_diseases)))
    print('Node dimension : ', node_dim)

    if args.cuda == True:
        class_weight = torch.FloatTensor([1, 1]).cuda()  ##
    else:
        class_weight = torch.FloatTensor([1, 1])
    criterion = nn.NLLLoss(weight=class_weight)
    criterion_rp = Cluster_Loss(threshold=0.8, weight=0.0001,cuda=args.cuda)
    optimizer_lp_model = optim.Adam(list(gcn.parameters()) + list(lp_model.parameters()),lr=args.lr)

    loss = 0
    rp_matrices, lp_model_evals = [], []

    f = open(files_home + '/networks/adj_matrix_%d_full' % (args.number), 'wb')
    adj_full = trainset.get_full_adj_matrix(f_ggi, f_dds)
    cPickle.dump(adj_full, f)

    for epoch in range(0, args.epochs):  #

        running_loss = 0.0
        adj_matrix = trainset.get_adj_matrix(f_ggi, f_dds)

        adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix)
        if args.cuda == True:
            adj_matrix.cuda()
        lp_model.train()

        gcn.train()
        init_input = torch.LongTensor([j for j in range(0, node_count)])
        if args.cuda == True:
            init_input.cuda()
        rp_matrix = gcn(init_input, adj_matrix)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            input1, input2 = inputs[0], inputs[1]
            if args.cuda == True:
                inputs[0].cuda()
                inputs[1].cuda()

            labels = labels.view(-1)
            if args.cuda == True:
                labels.cuda()
            rps1 = F.embedding(input1, rp_matrix)
            rps2 = F.embedding(input2, rp_matrix)

            outputs = lp_model(rps1, rps2)
            loss = criterion(outputs, labels) + loss

            # '''
            if (i + 1) % 1000 == 0:
                update_sm = False
                if (i + 1) % 9000 == 0:
                    update_sm = True
                loss = loss + criterion_rp(gcn.embedding.weight, update_sm)
            # '''
            if (i + 1) % 100 == 0:
                optimizer_lp_model.zero_grad()
                loss.backward()
                optimizer_lp_model.step()
                running_loss += loss.item()
                if (i + 1) % 1000 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
                loss = 0
                init_input = torch.LongTensor([j for j in range(0, node_count)])
                if args.cuda == True:
                    init_input.cuda()
                rp_matrix = gcn(init_input, adj_matrix)

        if (epoch > args.epochs-5 and epoch < args.epochs):  #
            f = open(files_home + '/networks/adj_matrix_%d_%d' % (args.number, epoch), 'wb')
            adj_matrix = trainset.get_adj_matrix(f_ggi, f_dds)
            cPickle.dump(adj_matrix, f)

            torch.save(gcn.state_dict(), files_home + '/networks/GCN_%d_%d.pth' % (args.number, epoch))
            torch.save(lp_model.state_dict(), files_home + '/networks/Link_Prediction_%d_%d.pth' % (args.number, epoch))

        trainset.reassign_samples(gcn.embedding.weight)  #
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)  #

    endtime = datetime.now()
    print('finish train model! run spend ',endtime-starttime)

if __name__ == '__main__':
    files_home = files_name['file_home']
    main(files_home)
