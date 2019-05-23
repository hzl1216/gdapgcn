from torch.utils.data.dataset import Dataset
from dataset_tool import *
class Sample_Set(Dataset):
    '''
    train dataSet
    '''

    def __init__(self,file_train_sample, ggi,dds,node2index=None):
        self.pairs_train = get_pairs(file_train_sample)
        self.all_genes, self.all_diseases = get_items(self.pairs_train)
        self.dropout = 0.5
        self.nega_weight = 10
        self.positive_samples_target, self.positive_samples_base = [], []
        if node2index is None:
            self.node2index = assign_index(self.all_genes, self.all_diseases)
        else:
            self.node2index = node2index
        self.added_samples = set()
        self.ggi=ggi
        self.dds=dds

    def reassign_samples(self):
        '''
        renew generate positive samples and negative samples
        :return:
        '''
        self.positive_samples_target, self.positive_samples_base = [], []
        for pair in list(self.pairs_train) + list(self.added_samples):
            if random.uniform(0, 1) < self.dropout:
                self.positive_samples_target.append(pair)
            else:
                self.positive_samples_base.append(pair)

        self.adj_matrix = adjacency_matrix(self.positive_samples_base, self.ggi, self.dds, self.node2index)
        self.negative_samples = generate_negative(self.all_genes, self.all_diseases, self.positive_samples_target,
                                                  self.nega_weight,self.pairs_train,self.node2index)
        self.samples = merge_samples(self.nega_weight, self.positive_samples_target, self.negative_samples)

    def add_samples(self, new_samples):
        '''
        add samples to dataSet, is external's pairs
        :param new_samples: new_samples
        :return:
        '''
        self.added_samples = self.added_samples | new_samples

    def remove_samples(self, samples):
        '''
        remove samples from external's pairs
        :param del_samples:
        :return:
        '''
        self.added_samples = self.added_samples - samples

    def get_adj_matrix(self):
        '''
        get positive_samples_target adjacency matrix
        :return:adjacency matrix ,type is coo_matrix
        '''
        return self.adj_matrix

    def get_full_adj_matrix(self):
        '''
        get all positive samples adjacency matrix
        :return: adjacency matrix ,type is coo_matrix
        '''
        return adjacency_matrix(self.positive_samples_base + self.positive_samples_target, self.ggi, self.dds,
                                                      self.node2index)


    def get_node2index(self):
        '''
        get node's index
        :return: node's index ,type is dict
        '''
        return self.node2index

    def get_added_item(self, batchsize):
        batch_list, batch1, batch2 = [], [], []
        for a_s in self.added_samples:
            as1, as2 = node_vector(a_s, self.node2index)
            batch1.append(as1)
            batch2.append(as2)
            if len(batch1) == batchsize:
                batch_list.append((torch.stack(batch1, 0), torch.stack(batch2, 0)))
                batch1, batch2 = [], []
        return batch_list

    def __getitem__(self, index):
        if index % (1 + self.nega_weight) == 0:
            target_flag = 1
        else:
            target_flag = 0
        item = self.samples[index]
        sample = node_vector(item, self.node2index)
        target = target_vector(target_flag)
        return sample, target

    def __len__(self):
        return len(self.samples)


class Sample_Set_Test(Dataset):
    '''
    test dataSet
    '''

    def __init__(self, file_test_sample,file_train, node2index, nega_weight=10):

        pairs_test = get_pairs(file_test_sample)
        pairs_train = get_pairs(file_train)
        all_genes, all_diseases = get_items(pairs_train)
        self.nega_weight = nega_weight
        self.positive_samples = list(pairs_test)
        self.negative_samples = generate_negative(all_genes, all_diseases, self.positive_samples, self.nega_weight,pairs_train|pairs_test,node2index)
        self.samples = merge_samples(self.nega_weight, self.positive_samples, self.negative_samples)
        self.node2index = node2index

    def __getitem__(self, index):
        if index % (1 + self.nega_weight) == 0:
            target_flag = 1
        else:
            target_flag = 0
        item = self.samples[index]
        sample = node_vector(item, self.node2index)
        target = target_vector(target_flag)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_item_by_ids(self, id1, id2):
        sample = node_vector((id1, id2), self.node2index)
        return sample







