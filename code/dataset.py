from torch.utils.data.dataset import Dataset
from dataset_tool import *
import torch.nn.functional as F
class Sample_Set(Dataset):

    def __init__(self, file_train_sample, node2index=None):

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
        self.positive_samples = None

    def reassign_samples(self, embedding):
        self.embedding = embedding
        self.positive_samples_target, self.positive_samples_base = [], []
        for pair in list(self.pairs_train) + list(self.added_samples):
            if random.uniform(0, 1) < self.dropout:
                self.positive_samples_target.append(pair)
            else:
                self.positive_samples_base.append(pair)

        self.positive_samples = self.positive_samples_base + self.positive_samples_target
        self.negative_samples = generate_negative(self.all_genes, self.all_diseases, self.positive_samples,
                                                  self.nega_weight)
        self.samples = merge_samples(self.nega_weight, self.positive_samples_target, self.negative_samples)

    def add_samples(self, new_samples):
        self.added_samples = self.added_samples | new_samples

    def remove_samples(self, del_samples):
        self.added_samples = self.added_samples - del_samples

    def get_adj_matrix(self, ggi, dds):
        adj_matrix = adjacency_matrix(self.positive_samples_base, ggi, dds, self.node2index, 0.0)
        return adj_matrix

    def get_full_adj_matrix(self, ggi, dds):
        adj_matrix = adjacency_matrix(self.positive_samples_base + self.positive_samples_target, ggi, dds,
                                      self.node2index, self.dropout)
        return adj_matrix

    def get_node2index(self):
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

    def __init__(self, file_test_sample, node2index, nega_weight=10):

        pairs_test = get_pairs(file_test_sample)
        test_genes, test_diseases = get_items(pairs_test)
        self.nega_weight = nega_weight
        self.positive_samples = list(pairs_test)
        self.negative_samples = generate_negative(test_genes, test_diseases, self.positive_samples, self.nega_weight)
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

class Sample_Set_Top_set(Dataset):
    def __init__(self, file_train, file_test,node2index,load_rate=1):
        self.get_rank_test_samples(file_train, file_test)
        self.node2index = node2index
        self.samples = self.build_samples(load_rate)


    """ Construct test verification set"""
    def get_rank_test_samples(self, f_train, f_test):
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

        self.dis2gene_test_true = dis2gene_test_true
        self.dis2gene_test_all = dis2gene_test_all

    def build_samples(self,load_rate):
        samples = []
        for d in self.dis2gene_test_all.keys():

            if random.uniform(0, 1) <= load_rate:
                continue  # only test to a part, for speed up
            # gene set to be predicted by the disease
            d_genes = list(self.dis2gene_test_all[d])
            for gene in d_genes:
                sample = (str(gene), d)
                samples.append(sample)
        return samples

    def __getitem__(self, index):
        if self.samples[index][1] in self.dis2gene_test_true[self.samples[index][1]]:
            target = 1
        else:
            target = 0
        item = self.samples[index]
        sample = node_vector(item, self.node2index)
        return sample, target

    def __len__(self):
        return len(self.samples)






