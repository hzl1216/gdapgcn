import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class Cluster_Loss(_Loss):
    def __init__(self, threshold, weight, cuda):
        super(Cluster_Loss, self).__init__()
        self.threshold = threshold
        self.cuda = cuda
        self.weight = weight
        self.avg_rp = None

    def get_avg_rp(self,rp):
        rp_data = rp.data
        '''
        # another simple way to compute cos_sim, but need too large GPU space
        rp_dot = torch.mm(rp_data, torch.t(rp_data)) # Rij = sum(Xik*Yjk for k)
        rp_square = torch.mul(rp_data, rp_data) # Rij = Xij*Xij
        rp_square_sum = torch.sum(rp_square,1) # Ri = sum(Xij for j)
        rp_l2 = torch.sqrt(rp_square_sum) # Ri = sqrt(Ri)
        rp_l2 = torch.unsqueeze(rp_l2,0) # [[R1, R2, ..., Ri]]
        rp_l2_dot = torch.mm(torch.t(rp_l2),rp_l2) # Rij = Ri*Rj
        rp_bottom = torch.tensor([[1e-8]]).to(self.device) # .expand(rp_l2_dot.size())
        rp_l2_dot = torch.max(rp_l2_dot,rp_bottom)
        cos_sm = torch.div(rp_dot,rp_l2_dot)
        '''
        cos_sm = []
        for i in range(0,rp_data.size()[0]):
            rp1 = torch.unsqueeze(rp_data[i],0)
            column = F.cosine_similarity(rp1,rp_data)
            cos_sm.append(column)
        cos_sm = torch.stack(cos_sm,0)

        threshold = torch.tensor([[self.threshold]])
        if self.cuda == True:
            threshold.cuda() # .expand(cos_sm.size())
        bit_sm = cos_sm > threshold # Fullfil with 0 or 1

        index1_list, index2_list, value_list = [],[],[]
        row_sum = []
        for i in range(0,bit_sm.size()[0]):
            row_sp = bit_sm[i].to_sparse()
            index1 = row_sp.indices()
            index2 = torch.LongTensor([[i]]).expand(index1.size())
            if self.cuda == True:
                index2.cuda()
            value = row_sp.values()
            row_sum.append(value.size()[0])
            index1_list.append(index1)
            index2_list.append(index2)
            value_list.append(value)
        indices1 = torch.cat(index1_list,1)
        indices2 = torch.cat(index2_list,1)
        indices = torch.cat([indices1,indices2])
        values = torch.cat(value_list)
        print(values.size()) # the number of similarity links
        sparse_sm = torch.sparse.FloatTensor(indices, values.float(), bit_sm.size())

        sm_rowsum_recip = torch.reciprocal(torch.FloatTensor(row_sum))
        if self.cuda == True:
            sm_rowsum_recip.cuda()
                # Si = 1/(sum(Sij) for j)
        sm_rowsum_recip = torch.unsqueeze(sm_rowsum_recip, 1) # [[S1], [S2], ..., [Si]]
        sm_rowsum_recip = sm_rowsum_recip.expand(rp.size()) # [[S1,S1,..], [S2,S2,..], ..., [Si,Si,..]]
        sum_rp = torch.spmm(sparse_sm,rp_data)
        avg_rp = torch.mul(sum_rp,sm_rowsum_recip)

        return avg_rp

    def forward(self, rp, update):
        if update or (self.avg_rp is None):
            self.avg_rp = self.get_avg_rp(rp)
        diff_rp = rp - self.avg_rp
        loss_rp = torch.pow(diff_rp,2)
        print('Cluster Loss:', self.weight*torch.sum(loss_rp).item())
        return self.weight*torch.sum(loss_rp)

