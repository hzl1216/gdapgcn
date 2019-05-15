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
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--load_node_index', type=ast.literal_eval, default=True, dest='load_node_index',
                            help='load node index,(default: True, False meaning renew node index),'
                                                     'input should be either "True" or "False".')
parser.add_argument('--number', type=int, default=1,
                            help='number to use sign the different train model')
parser.add_argument('--load_model', type=ast.literal_eval, default=True, dest='load_model',
                                    help='load model continue train,(default: True, False meaning renew node index),'
                                        'input should be either "True" or "False".')

args = parser.parse_args()

def main(files_home):
    starttime = datetime.now()
    print('start train model ',starttime)
#    np.random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    if args.cuda==True:
#        torch.cuda.manual_seed(args.seed)
    # Load data
    f_ggi = os.path.join(files_home, files_name['graph_ggn_file'])
    f_dds = os.path.join(files_home, files_name['graph_dds_file'])

    f_train = os.path.join(files_home, files_name['train_file'])
    f_test = os.path.join(files_home, files_name['test_file'])

    node_index_file = os.path.join(files_home, files_name['node_index'])
    ### load node_index
    if args.load_node_index==True:
        try:
            print('load old node_index')
            f = open(node_index_file, 'rb')
            node2index = cPickle.load(f)
            trainset = Sample_Set(f_train,f_ggi,f_dds, node2index)
        except:
            ### new node_index
            print('load node_index failed')
            trainset = Sample_Set(f_train,f_ggi,f_dds)
            node2index = trainset.get_node2index()
            f = open(node_index_file,'wb')
            cPickle.dump(node2index,f)

    else:
        ### new node_index
        print('new node_index')
        trainset = Sample_Set(f_train,f_ggi,f_dds)
        node2index = trainset.get_node2index()
        f = open(node_index_file,'wb')
        cPickle.dump(node2index,f)

    node_count = len(node2index)
    node_dim = 128
    n_repr = 128
    gcn = GCN(node_count,node_dim,n_repr,dropout=args.dropout)
    lp_model = Link_Prediction(n_repr,dropout=args.dropout)

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

    f = open(files_home + '/networks/adj_matrix_%d_full' % (args.number), 'wb')
    adj_full = trainset.get_full_adj_matrix()
    cPickle.dump(adj_full, f)

    # load the last model
    number=args.number
    last_epoch=0
    if args.load_model ==True: 
        print('load model,continue train')
        files= os.listdir(files_home + '/networks/') #得到文件夹下的所有文件名称
        s = []
        for file in files:
            if 'Link_Prediction_%d_'%(number) in file :
                s.append(int(file[len('Link_Prediction_%d_'%(number)):].split('.')[0]))
        if len(s)>0:                                                                        
            last_epoch = max(s)                                                                 
            gcn.load_state_dict(torch.load(files_home + '/networks/GCN_%d_%d.pth'%(number,last_epoch)))
            lp_model.load_state_dict(torch.load(files_home + '/networks/Link_Prediction_%d_%d.pth'%(number,last_epoch )))
            last_epoch+=1
        else:
             last_epoch=0
    else:
        print('retrain model')
        files= os.listdir(files_home + '/networks/') #得到文件夹下的所有文件名称
        for file in files:
            if 'Link_Prediction_%d_'%(number) in file or  'GCN_%d_'%(number) in file:
                os.remove(files_home + '/networks/'+file) 

    running_losses = []
    cluster_losses = []
    for epoch in tqdm(range(last_epoch,args.epochs)):  #

        running_loss = 0.0
        adj_matrix = trainset.get_adj_matrix()

        if args.cuda == True:
            adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix).cuda()
        else:
            adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix)
        lp_model.train()

        gcn.train()
        if args.cuda == True:
            init_input = torch.LongTensor([j for j in range(0, node_count)]).cuda()
        else:
            init_input = torch.LongTensor([j for j in range(0, node_count)])
        rp_matrix = gcn(init_input, adj_matrix)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if args.cuda == True:
                input1, input2 = inputs[0].cuda(), inputs[1].cuda()
            else:
                input1, input2 = inputs[0], inputs[1]
            if args.cuda == True:
                labels = labels.view(-1).cuda()
            else:
                labels = labels.view(-1)
            rps1 = F.embedding(input1, rp_matrix)
            rps2 = F.embedding(input2, rp_matrix)

            outputs = lp_model(rps1, rps2)
            loss = criterion(outputs, labels) + loss

            # '''
            if (i+1) % 1000 == 0:
                update_sm = False
                if i+1 % 9000 == 0:
                    update_sm = True
                cluster_loss = criterion_rp(gcn.embedding.weight, update_sm)
                cluster_losses.append(cluster_loss)
                loss+=cluster_loss
            # '''
            if (i+1)  % 100 == 0:
                optimizer_lp_model.zero_grad()
                loss.backward()
                optimizer_lp_model.step()
                running_loss += loss.item()
                if (i+1) % 1000 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch, i+1, running_loss))
                running_losses.append(running_loss)
                running_loss = 0.0
                loss = 0
                rp_matrix = gcn(init_input, adj_matrix)

        if (epoch >= args.epochs-5 and epoch < args.epochs):  #
            torch.save(gcn.state_dict(), files_home + '/networks/GCN_%d_%d.pth' % (number, epoch))
            torch.save(lp_model.state_dict(), files_home + '/networks/Link_Prediction_%d_%d.pth' % (number, epoch))
        elif epoch%9==0:
            torch.save(gcn.state_dict(), files_home + '/networks/GCN_%d_%d.pth'%(number,epoch))
            torch.save(lp_model.state_dict(), files_home + '/networks/Link_Prediction_%d_%d.pth'%(number,epoch))
        print('reset train samples set')
        trainset.reassign_samples(gcn.embedding.weight)  #
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)  #
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x1 =  np.arange(0, args.epochs,args.epochs/len(running_losses))
    x2 =  np.arange(0, args.epochs,args.epochs/len(cluster_losses))
    y1 = running_losses
    y2=  cluster_losses
    ax1.plot(x1, y1, 'g-')
    ax2.plot(x2, y2, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Total loss', color='g')
    ax2.set_ylabel('Cluster_loss', color='b')

    plt.savefig(files_home + '/results/loss_%d.png'%number)
    plt.close()
    endtime = datetime.now()
    print('finish train model! run spend ',endtime-starttime)

if __name__ == '__main__':
    files_home = files_name['file_home']
    main(files_home)
