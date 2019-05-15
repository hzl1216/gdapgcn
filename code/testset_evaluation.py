import os
import csv
import torch
import _pickle as cPickle
import pandas as pd
import torch.nn.functional as F
from pandas.io.parsers import read_csv
from torch.utils.data import DataLoader
from model import GCN, Link_Prediction
from utils import sparse_mx_to_torch_sparse_tensor,evaluation
from dataset import Sample_Set_Test, node_vector
from train import args
import torch.nn as nn
from file_name import files_name
from datetime import datetime
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score as auc
""" Construct test verification set"""



def get_rank_test_samples(f_train, f_test):
    '''
    :param f_train: the gene and disease associations in train set
    :param f_test: the gene and disease associations in test set
    :return: the all set of the gene and disease associations, the gene that in test set but not in train set ,
    use those genes to predict disease associations
    '''
    all_genes = set()
    dis2gene_train = dict()
    dis2gene_test_true = dict()
    dis2gene_test_all = dict()

    df = read_csv(f_train)
    for i in range(0, len(df)):
        gid = str(df['gid'][i])
        did = df['did'][i]
        if did not in dis2gene_train.keys():
            dis2gene_train[did] = set()
        dis2gene_train[did].add(gid)

    df = read_csv(f_test)
    for i in range(0, len(df)):
        gid = str(df['gid'][i])
        did = df['did'][i]
        all_genes.add(gid)
        if did not in dis2gene_test_true.keys():
            dis2gene_test_true[did] = set()
        dis2gene_test_true[did].add(gid)

    for key in dis2gene_test_true.keys():
        dis2gene_test_all[key] = all_genes - dis2gene_train[key]

    return dis2gene_test_true, dis2gene_test_all


def get_test_prf(testloader,rp_matrix, lp):
    tp, fp, tn, fn = 0, 0, 0, 0
    result = []
    device = torch.device('cuda:0')
    # f_curve = open(path_ + '/results/curve.txt','w')

    for i, data in enumerate(testloader, 0):


        inputs, labels = data
        input1, input2 = inputs[0].to(device), inputs[1].to(device)
        labels = labels.view(-1).to(device)

        rp1 = F.embedding(input1, rp_matrix)
        rp2 = F.embedding(input2, rp_matrix)

        outputs = lp(rp1, rp2)

        for j in range(0, outputs.size()[0]):
            id1, id2 = input1[j].item(), input2[j].item()
            if outputs.data[j][1].item() > outputs.data[j][0].item():
                if labels.data[j].item() == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if labels.data[j].item() == 1:
                    fn = fn + 1
                else:
                    tn = tn + 1

            result.append((labels.data[j].item(), outputs.data[j][1].item()))
            # f_curve.write(str(labels.data[j].item())+' '+str(outputs.data[j][1].item())+'\n')

    result_sorted = sorted(result, key=lambda x: x[0], reverse=True)
    y_label = [r[0] for r in result_sorted]
    y_score = [r[1] for r in result_sorted]

    prec = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1score = 2 * prec * recall / (prec + recall)
    auc_score = auc(y_label, y_score)
    # f_curve.close()
    print('prec %f, recall %f , f1score %f, auc_score %f'%(prec, recall, f1score, auc_score))
    return prec, recall, f1score, auc_score

def get_test_priorization_gcn(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp, total_top=10, batch_size=1024):
    '''
    :param dis2gene_test_true: the validate set
    :param dis2gene_test_all: the gene set of waiting for prediction
    :param feature_matrix: feature matrix
    :param lp:the model of link prediction
    :return: performances
    '''
    if args.cuda == True:
        lp = nn.DataParallel(lp)
        lp.cuda()
    tp, fp, fn = [[] for i in range(0, total_top)], [[] for i in range(0, total_top)], [[] for i in range(0, total_top)]
    ap_numerator, ap_denumerator = 0.0, 0.0
    count = 0
    path_ = files_name['file_home']

    # save the prediction result
    number = args.number
    gp_file = path_ + '/results/gene_priorization_%d.csv' % number
    df_gp = pd.DataFrame(columns=('did', 'tp', 'fp', 'fn', 'gid_predict', 'gid_real'))
    df_gp.to_csv(gp_file)
    csvfile_gp = open(gp_file, 'a')
    writer_gp = csv.writer(csvfile_gp)

    f = open(path_ + '/node_index', 'rb')
    node2index = cPickle.load(f)

    lp.eval()

    for d in dis2gene_test_all.keys():

        # gene set to be predicted by the disease
        d_genes = list(dis2gene_test_all[d])
        result = dict()
        #
        for k in range(0, len(d_genes) // batch_size + 1):
            base = k * batch_size
            if base == len(d_genes):
                break

            inputs_d, inputs_g = [], []
            for i in range(base, min(base + batch_size, len(d_genes))):
                g = d_genes[i]
                input_g, input_d = node_vector((str(g), d), node2index)
                inputs_g.append(input_g)
                inputs_d.append(input_d)
            inputs_g = torch.stack(inputs_g).cuda()
            inputs_d = torch.stack(inputs_d).cuda()

            rp1 = F.embedding(inputs_g, feature_matrix)  # g is fronter than d in pairs
            rp2 = F.embedding(inputs_d, feature_matrix)
            outputs = lp(rp1, rp2)

            for j in range(0, outputs.size()[0]):
                result[d_genes[base + j]] = outputs.data[j][1].item()
        result = sorted(result.items(), key=lambda d: d[1], reverse=True)
        k = len(dis2gene_test_true[d])
        top_list = set([result[i][0] for i in range(0, k)])
        ap_numerator = ap_numerator + len(top_list & dis2gene_test_true[d])
        ap_denumerator = ap_denumerator + k

        for top in range(1, total_top + 1):
            top_list = set([result[i][0] for i in range(0, top)])
            tp[top - 1].append(len(top_list & dis2gene_test_true[d]))
            fp[top - 1].append(len(top_list - dis2gene_test_true[d]))
            fn[top - 1].append(len(dis2gene_test_true[d] - top_list))

        writer_gp.writerow(
            ['', d, sum(tp[total_top - 1]), sum(fp[total_top - 1]), sum(fn[total_top - 1]), str(top_list),
             str(dis2gene_test_true[d])])
        writer_gp.writerow(['', '', '', '', '', str([result[i][1] for i in range(0, total_top)]), ''])

    prec_mean, recall_mean, f1score_mean = evaluation(tp,fp,fn,total_top)
    ap = ap_numerator / ap_denumerator
    return ap, prec_mean, recall_mean, f1score_mean

def get_test_priorization_word(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp, total_top=10, batch_size=256):
    '''
    :param dis2gene_test_true: the validate set
    :param dis2gene_test_all: the gene set of waiting for prediction
    :param feature_matrix: feature matrix
    :param lp:the model of link prediction
    :return: performances
    '''
    tp, fp, fn = [[] for i in range(0, total_top)], [[] for i in range(0, total_top)], [[] for i in range(0, total_top)]
    ap_numerator, ap_denumerator = 0.0, 0.0
    count = 0
    path_ = files_name['file_home']
    vec_dir = os.path.join(path_, files_name['word2vec'])
    my_word2vec = Word2Vec.load(vec_dir)
    # save the prediction result
    number = args.number
    gp_file = path_ + '/results/gene_priorization_%d.csv' % number
    df_gp = pd.DataFrame(columns=('did', 'tp', 'fp', 'fn', 'gid_predict', 'gid_real'))
    df_gp.to_csv(gp_file)
    csvfile_gp = open(gp_file, 'a')
    writer_gp = csv.writer(csvfile_gp)

    for d in dis2gene_test_all.keys():

        # gene set to be predicted by the disease
        d_genes = list(dis2gene_test_all[d])
        #
        result = []
        results = my_word2vec.predict_output_word([d],topn=3000)
        for r in results:
            if r[0] in d_genes:
                result.append(r)
        k = len(dis2gene_test_true[d])
        top_list = set([result[i][0] for i in range(0, k)])
        ap_numerator = ap_numerator + len(top_list & dis2gene_test_true[d])
        ap_denumerator = ap_denumerator + k

        for top in range(1, total_top + 1):
            top_list = set([result[i][0] for i in range(0, top)])
            tp[top - 1].append(len(top_list & dis2gene_test_true[d]))
            fp[top - 1].append(len(top_list - dis2gene_test_true[d]))
            fn[top - 1].append(len(dis2gene_test_true[d] - top_list))

        writer_gp.writerow(
            ['', d, sum(tp[total_top - 1]), sum(fp[total_top - 1]), sum(fn[total_top - 1]), str(top_list),
             str(dis2gene_test_true[d])])
        writer_gp.writerow(['', '', '', '', '', str([result[i][1] for i in range(0, total_top)]), ''])

    prec_mean, recall_mean, f1score_mean = evaluation(tp,fp,fn,total_top)
    ap = ap_numerator / ap_denumerator
    return ap, prec_mean, recall_mean, f1score_mean



def get_test_priorization_word_gcn(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp, total_top=10, batch_size=1024,threshold=0.8):
    '''
    :param dis2gene_test_true: the validate set
    :param dis2gene_test_all: the gene set of waiting for prediction
    :param feature_matrix: feature matrix
    :param lp:the model of link prediction
    :return: performances
    '''
    files_home = files_name['file_home']
    if args.cuda == True:
        lp = nn.DataParallel(lp)
        lp.cuda()

    vec_dir = os.path.join(files_home, files_name['word2vec'])
    my_word2vec = Word2Vec.load(vec_dir)

    # save the prediction result
    number = args.number
    gp_file = files_home + '/results/gene_priorization_%d.csv' % number
    df_gp = pd.DataFrame(columns=('did', 'tp', 'fp', 'fn', 'gid_predict', 'gid_real'))
    df_gp.to_csv(gp_file)
    csvfile_gp = open(gp_file, 'a')
    writer_gp = csv.writer(csvfile_gp)

    f = open(os.path.join(files_home, files_name['node_index']), 'rb')
    node2index = cPickle.load(f)
    node2index_r = dict()
    for key in node2index.keys():
        value = node2index[key]
        node2index_r[value] = key

    lp.eval()
    gene_end = 8948  # nodex2index 0-8948 is gene
    genes_tensor = [torch.tensor(i).cuda() for i in range(0, gene_end)]
    genes_tensor = torch.stack(genes_tensor)
    rp_genes = F.embedding(genes_tensor, feature_matrix)

    tp, fp, fn = [[] for i in range(0, total_top)], [[] for i in range(0, total_top)], [[] for i in range(0, total_top)]
    ap_numerator, ap_denumerator = 0.0, 0.0
    #
    for d in dis2gene_test_all.keys():

        d_index = node2index[d]
        disease_tensor = [torch.tensor(d_index).cuda() for i in range(0, batch_size)]
        disease_tensor = torch.stack(disease_tensor)
        rp_disease = F.embedding(disease_tensor, feature_matrix)

        result_vector = []
        # batch calculation
        for k in range(0, gene_end // batch_size + 1):
            base = k * batch_size
            if base == gene_end:
                break
            rp_g = rp_genes[base:min(base + batch_size, gene_end)]
            rp_d = rp_disease[0:rp_g.size()[0]]
            outputs = lp(rp_g, rp_d)
            outputs = torch.exp(outputs)
            result_vector.extend(outputs.data[:, 1].tolist())

        # get candidate gene set by the gene the lp output above threshold
        g_candidate = dict()
        for j in range(0, gene_end):
            g_name = node2index_r[j]
            if g_name in dis2gene_test_all[d]:
                g_candidate[g_name] = result_vector[j]

        g_candidate = sorted(g_candidate.items(), key=lambda d: d[1], reverse=True)
        tmp_i = total_top
        while g_candidate[tmp_i][1] > threshold:
            tmp_i = tmp_i + 1

        g_candidate = set([g_candidate[j][0] for j in range(0, max(tmp_i, len(dis2gene_test_true[d])))])  # tmp_i

        # get gene sort result by word2vec Genetic sequencing
        result = []
        results = my_word2vec.predict_output_word([d],topn=20000)
        for r in results:
            if r[0] in g_candidate:
                result.append(r)

        k = len(dis2gene_test_true[d])
        top_list = set([result[j][0] for j in range(0, k)])
        ap_numerator = ap_numerator + len(top_list & dis2gene_test_true[d])
        ap_denumerator = ap_denumerator + k
        for top in range(1, total_top + 1):
            top_list = set([result[j][0] for j in range(0, top)])
            tp[top - 1].append(len(top_list & dis2gene_test_true[d]))
            fp[top - 1].append(len(top_list - dis2gene_test_true[d]))
            fn[top - 1].append(len(dis2gene_test_true[d] - top_list))
        writer_gp.writerow(
            ['', d, sum(tp[total_top - 1]), sum(fp[total_top - 1]), sum(fn[total_top - 1]), str(top_list),
             str(dis2gene_test_true[d])])
        writer_gp.writerow(['', '', '', '', '', str([result[i][1] for i in range(0, total_top)]), ''])

    prec_mean, recall_mean, f1score_mean = evaluation(tp,fp,fn,total_top)
    ap = ap_numerator / ap_denumerator
    return ap, prec_mean, recall_mean, f1score_mean
def main(files_home):
    starttime = datetime.now()
    print('start test model ', starttime)

    number = args.number

    f = open(os.path.join(files_home, files_name['node_index']), 'rb')
    node2index = cPickle.load(f)

    f_train = os.path.join(files_home, files_name['train_file'])
    f_test = os.path.join(files_home, files_name['test_file'])  ###

    testset = Sample_Set_Test(f_test,f_train, node2index)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    node_count = len(node2index)
    node_dim = 128
    n_repr = 128
    gcn = GCN(node_count,node_dim,n_repr,dropout=args.dropout)
    lp = Link_Prediction(n_repr,dropout=args.dropout)
    if args.cuda == True:
        gcn.cuda()
        lp = nn.DataParallel(lp)
        lp.cuda()

    init_input = torch.LongTensor([j for j in range(0, node_count)]).cuda()

    dis2gene_test_true, dis2gene_test_all = get_rank_test_samples(f_train, f_test)

    f = open(files_home + '/networks/adj_matrix_%d_full' % (number), 'rb')
    full_adj_matrix = cPickle.load(f)
    full_adj_matrix = sparse_mx_to_torch_sparse_tensor(full_adj_matrix).cuda()

    for epoch in tqdm(range(54, args.epochs)):
        if epoch%9!=0 and epoch<args.epochs-5:
            continue
        gcn.load_state_dict(torch.load(files_home + '/networks/GCN_%d_%d.pth'%(number,epoch)))
        lp.load_state_dict(torch.load(files_home + '/networks/Link_Prediction_%d_%d.pth'%(number,epoch)))
        gcn.eval()

        feature_matrix = gcn(init_input, full_adj_matrix)
        get_test_prf(testloader, feature_matrix, lp)

        if 0:
            # no use similarity matrix to solve
            print('no use similarity matrix to solve')
            ap, prec, recall, f1score = get_test_priorization_gcn(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp)
            #  use similarity matrix to iterative solution
        else:
            print('use similarity matrix to iterative solution')
            ap, prec, recall, f1score = get_test_priorization_word_gcn(dis2gene_test_true, dis2gene_test_all, feature_matrix,
                                                                  lp)
        print('Performance for number=%d epoch=%d' % (number, epoch))
        print('AP: ', ap)
        print('Prec: ', prec)
        print('Recall: ', recall)
        print('F1score: ', f1score)

    endtime = datetime.now()
    print('finish test model! run spend ', endtime - starttime)


if __name__ == '__main__':
    files_home = files_name['file_home']
    main(files_home)
