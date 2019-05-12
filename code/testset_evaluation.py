import os
import csv
import torch
import _pickle as cPickle
import pandas as pd
import torch.nn.functional as F
from pandas.io.parsers import read_csv
from torch.utils.data import DataLoader
from model import GCN, Link_Prediction
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import Sample_Set_Test, node_vector
from train import args
import torch.nn as nn
from file_name import files_name
from datetime import datetime

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


def get_test_priorization(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp, total_top=10, batch_size=256):
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

    # csvfile_gp.close()
    sample_length = len(tp[0])
    prec = [[float(tp[top][i]) / (tp[top][i] + fp[top][i]) for i in range(0, sample_length)] for top in
            range(0, total_top)]
    recall = [[float(tp[top][i]) / (tp[top][i] + fn[top][i]) for i in range(0, sample_length)] for top in
              range(0, total_top)]
    f1score = []
    for top in range(0, total_top):
        f1score.append([])
        for i in range(0, sample_length):
            if prec[top][i] + recall[top][i] == 0:
                f1score[top].append(0)
            else:
                f1score[top].append(2 * prec[top][i] * recall[top][i] / (prec[top][i] + recall[top][i]))
    prec_mean = [sum(prec[top]) / sample_length for top in range(0, total_top)]
    recall_mean = [sum(recall[top]) / sample_length for top in range(0, total_top)]
    f1score_mean = [sum(f1score[top]) / sample_length for top in range(0, total_top)]
    ap = ap_numerator / ap_denumerator
    return ap, prec_mean, recall_mean, f1score_mean


def get_test_priorization_new(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp, total_top=10, batch_size=1024,
                              theta=0.8, threshold=0.8):
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

    result_tensor = []
    result_tensor_list = []

    lp.eval()
    gene_end = 8948  # nodex2index 0-8948 is gene
    genes_tensor = [torch.tensor(i).cuda() for i in range(0, gene_end)]
    genes_tensor = torch.stack(genes_tensor)
    rp_genes = F.embedding(genes_tensor, feature_matrix)

    count = 0
    for d in dis2gene_test_all.keys():

        if (count != 0) and (count % batch_size == 0):
            result_tensor = torch.tensor(result_tensor).cuda()
            # result_tensor = torch.exp(result_tensor)
            result_tensor_list.append(result_tensor)
            result_tensor = []

        d_index = node2index[d]
        disease_tensor = [torch.tensor(d_index).cuda() for i in range(0, batch_size)]
        disease_tensor = torch.stack(disease_tensor)
        rp_disease = F.embedding(disease_tensor, feature_matrix)

        result_vector = []
        for k in range(0, gene_end // batch_size + 1):
            base = k * batch_size
            if base == gene_end:
                break
            rp_g = rp_genes[base:min(base + batch_size, gene_end)]
            rp_d = rp_disease[0:rp_g.size()[0]]
            outputs = lp(rp_g, rp_d)
            outputs = torch.exp(outputs)
            result_vector.extend(outputs.data[:, 1].tolist())

        result_tensor.append(result_vector)
        count = count + 1

    base = 0
    tp, fp, fn = [[] for i in range(0, total_top)], [[] for i in range(0, total_top)], [[] for i in range(0, total_top)]
    ap_numerator, ap_denumerator = 0.0, 0.0
    f = open(os.path.join(files_home, files_name['similarity_matrix']), 'rb')
    adj_sparse = cPickle.load(f)
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj_sparse).cuda()
    adj_dense = adj_sparse.to_dense()

    for i in range(0, len(result_tensor_list)):
        result_tensor = result_tensor_list[i]
        # Smooth association matrix init_sim
        init_sim = []
        for n in range(result_tensor.size()[0]):
            d = list(dis2gene_test_all.keys())[base + n]
            d_index = node2index[d]
            init_sim.append(adj_dense[d_index, :])
        init_sim = torch.stack(init_sim)

        init_sim = torch.t(init_sim)
        for k in range(0, 1):
            init_sim = (1 - theta) * torch.spmm(adj_sparse, init_sim) + theta * init_sim  # +0.1*result_tensor_gcn
        init_sim = torch.t(init_sim)

        # Keep only the predicted results of the gene in dis2gene_test_all
        for n in range(0, init_sim.size()[0]):
            g_candidate_tensor = result_tensor_list[i][n]
            g_candidate = dict()
            d = list(dis2gene_test_all.keys())[base + n]
            for j in range(0, g_candidate_tensor.size()[0]):
                g_name = node2index_r[j]
                if g_name in dis2gene_test_all[d]:
                    g_candidate[g_name] = g_candidate_tensor.data[j].item()

            g_candidate = sorted(g_candidate.items(), key=lambda d: d[1], reverse=True)
            tmp_i = total_top
            while g_candidate[tmp_i][1] > threshold:
                tmp_i = tmp_i + 1

            g_candidate = set([g_candidate[j][0] for j in range(0, max(tmp_i, len(dis2gene_test_true[d])))])  # tmp_i

            result = dict()
            outputs = init_sim[n]
            d = list(dis2gene_test_all.keys())[base + n]
            for j in range(0, outputs.size()[0]):
                g_name = node2index_r[j]
                if g_name in g_candidate:
                    result[g_name] = outputs.data[j].item()

            result = sorted(result.items(), key=lambda d: d[1], reverse=True)

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

        base = base + init_sim.size()[0]

    sample_length = len(tp[0])
    prec = [[float(tp[top][i]) / (tp[top][i] + fp[top][i]) for i in range(0, sample_length)] for top in
            range(0, total_top)]
    recall = [[float(tp[top][i]) / (tp[top][i] + fn[top][i]) for i in range(0, sample_length)] for top in
              range(0, total_top)]
    f1score = []
    for top in range(0, total_top):
        f1score.append([])
        for i in range(0, sample_length):
            if prec[top][i] + recall[top][i] == 0:
                f1score[top].append(0)
            else:
                f1score[top].append(2 * prec[top][i] * recall[top][i] / (prec[top][i] + recall[top][i]))
    prec_mean = [sum(prec[top]) / sample_length for top in range(0, total_top)]
    recall_mean = [sum(recall[top]) / sample_length for top in range(0, total_top)]
    f1score_mean = [sum(f1score[top]) / sample_length for top in range(0, total_top)]
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

#    rp_matrices, lps = [], []
    for epoch in range(0, args.epochs):
        if epoch%9!=0 and epoch<args.epochs-5:
            continue
        if 0:
            gcn.load_state_dict(torch.load(files_home + '/networks/GCN_%d_%d.pth' % (number, epoch)))
            lp.load_state_dict(torch.load(files_home + '/networks/Link_Prediction_%d_%d.pth' % (number, epoch)))
        else:
            gcn.load_state_dict(torch.load(files_home + '/networks/GCN_last_%d.pth'%(epoch)))
            lp.load_state_dict(torch.load(files_home + '/networks/Link_Prediction_last_%d.pth'%(epoch)))
        gcn.eval()

        feature_matrix = gcn(init_input, full_adj_matrix)
#        rp_matrices.append(feature_matrix)
#        lps.append(lp)
        if 0:
            # no use similarity matrix to solve
            print('no use similarity matrix to solve')
            ap, prec, recall, f1score = get_test_priorization(dis2gene_test_true, dis2gene_test_all, feature_matrix, lp)
            print('Performance for number=%d epoch=%d' % (number, epoch))
            print('AP: ', ap)
            print('Prec: ', prec)
            print('Recall: ', recall)
            print('F1score: ', f1score)
            #  use similarity matrix to iterative solution
        else:
            print('use similarity matrix to iterative solution')
            ap, prec, recall, f1score = get_test_priorization_new(dis2gene_test_true, dis2gene_test_all, feature_matrix,
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
