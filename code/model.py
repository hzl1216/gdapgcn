from layers import *
import torch.nn.init as init


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, node_count, node_dim, n_representation,dropout=0.2):
        super(GCN, self).__init__()
        self.n_feature = node_dim
        self.n_hidden = 128
        self.n_representation = n_representation

        self.embedding = nn.Embedding(node_count, self.n_feature)
        self.gc1 = GraphConvolution(self.n_feature, self.n_hidden)
        self.gc2 = GraphConvolution(self.n_hidden, self.n_representation)
        self.dropout = nn.Dropout(p=0.2)
        self.tmp_linear = nn.Linear(self.n_feature, self.n_representation)
        self.init_weights()

    def forward(self, x, adj):  # , adj_g, adj_d
        x = self.embedding(x)  #

        x1 = F.relu(self.gc1(x, adj))
        x1 = self.gc2(x1, adj)

        x2 = self.dropout(x)
        x2 = self.tmp_linear(x2)
        out = x1 + x2  # + x2_g + x2_d # node_count * n_representation

        return out

    def init_weights(self):
        init.xavier_uniform_(self.tmp_linear.weight)
        init.xavier_uniform_(self.gc1.weight)
        init.xavier_uniform_(self.gc2.weight)


class Link_Prediction(nn.Module):
    def __init__(self, n_representation, hidden_dims=[128, 32], dropout=0.2):
        super(Link_Prediction, self).__init__()
        self.n_representation = n_representation
        self.linear1 = nn.Linear(2*self.n_representation, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], 2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weights()

    def forward(self, x1, x2):
        x = torch.cat((x1,x2),1) # N * (2 node_dim)
        
        x = F.relu(self.linear1(x)) # N * hidden1_dim
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # N * hidden2_dim
        x = self.dropout(x)
        x = self.linear3(x) # N * 2
        x = self.softmax(x) # N * ( probility of each event )
        return x

    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

