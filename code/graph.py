import random
import pandas as pd
import os
import csv
import time
from file_name import files_name
from utils import skip_gram
from datetime import datetime
homo_RATIO = 0.2 #
hetero_RATIO = 1 #


class Link(object):
    def __init__(self,link_to,link_weight,link_type):
        self.link_to = link_to
        self.link_weight = link_weight
        self.link_type = link_type

    """update link's weight, the function is not suggest be used in public """
    def __update_weight__(self, resize):
        self.link_weight = resize*self.link_weight


class Node(object):
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type
        self.links = []
        self.neighbors = set() # record in string

    """establish the association of the own node to the target node"""
    def set_link(self, target, score, type):
        if target.node_id in self.neighbors: # has recorded
            return False
        link = Link(target, score, type)
        self.links.append(link)
        self.neighbors.add(target.node_id)
        return True

    def has_link(self,dst):
        return dst in self.links

    """update link's weight,the function is suggest be used in public """
    def update_link_weight(self,link,resize):
        link.__update_weight__(resize)

    def link_weight_sum(self, type):
        weight_sum = 0.0
        for link in self.links:
            if link.link_type == type:
                weight_sum = weight_sum + link.link_weight
        return weight_sum

    def link_weight_normalization(self):
        '''
        normalize the weight of the links,to keep weight sum is GENE_RATIO+DISEASE_RATIO
        :return:
        '''
        homo_weight_sum = self.link_weight_sum('homo')
        hetero_weight_sum = self.link_weight_sum('hetero')

        if homo_weight_sum == 0.0:
            resize_hetero = (hetero_RATIO+homo_RATIO)/ hetero_weight_sum
        elif hetero_weight_sum == 0.0:
            resize_homo = (hetero_RATIO+homo_RATIO)/ homo_weight_sum
        else:
            resize_homo = homo_RATIO / homo_weight_sum
            resize_hetero = hetero_RATIO / hetero_weight_sum
        for link in self.links:
            if link.link_type == 'homo':
                link.__update_weight__(resize_homo)
            else:
                link.__update_weight__(resize_hetero)


    def random_next(self):
        '''
        random walk to next node
        :return:
        '''
        random_weight = random.uniform(0, homo_RATIO+hetero_RATIO)
        i = 0
        while (random_weight>=0.0)and(i<len(self.links)):
            random_weight = random_weight - self.links[i].link_weight
            i = i + 1
        if i == 0:
            i = 1
        return self.links[i-1].link_to

class Graph(object):
    def __init__(self):
        self.nodes = dict()
        self.disease_map = dict()
        self.nodes_count = 0
        self.links_count =0

    def get_size(self):
        return self.nodes_count, self.links_count

    def add_node(self,id , type):
        self.nodes[id] = Node(id, type)
        self.nodes_count = self.nodes_count + 1
        return self.nodes[id]

    def unifrom_graph(self):
        for node in self.nodes.values():
            node.link_weight_normalization()


    def load_disease_map(self, map_file):
        '''
        map key is MeshID,value is the set of diseaseIds
        :param map_file:
        :return:
        '''
        df_map = pd.read_csv(map_file)
        for i in range(0, len(df_map)):
            did = str(df_map.loc[i, 'diseaseId'])
            mid = str(df_map.loc[i, 'MeshID'])
            if mid not in self.disease_map.keys():
                self.disease_map[mid] = set()
            self.disease_map[mid].add(did)
        print('load disease map,have %d MeshID'%len(self.disease_map))

    def build_g_d_net(self,gda_file,train_file, test_file, test_ratio=0.1):
        '''
        build gene to disease sub net
        :param gda_file:  gene-disease association file
        :param train_file: train set file output
        :param test_file: test set file output
        :param test_ratio: the rate of test set in all set
        :return:
        '''
        c_source_set = set(['CTD_human', 'PSYGENET', 'HPO', 'ORPHANET', 'UNIPROT'])

        df_gda = pd.read_csv(gda_file, sep='\t', header=0,encoding = "ISO-8859-1")
        df_test = pd.DataFrame(columns=('gid', 'did', 'score'))
        df_test.to_csv(test_file)
        df_train = pd.DataFrame(columns=('gid', 'did', 'score'))
        df_train.to_csv(train_file)

        csvfile_test = open(test_file, 'a',newline ='')
        writer_test = csv.writer(csvfile_test)
        csvfile_train = open(train_file, 'a',newline ='')
        writer_train = csv.writer(csvfile_train)
        # load the gda files and construct g-d network or split into testset
        for i in range(0, len(df_gda)):
            gid = str(df_gda.loc[i, 'geneId'])
            did = str(df_gda.loc[i, 'diseaseId'])
            score = float(df_gda.loc[i, 'score'])
            source = df_gda.loc[i, 'source']
            source = set(source.split(';'))
            if len(source & c_source_set) == 0:
                continue
                # set as link or add to positive testset
            if random.uniform(0, 1) < test_ratio:
                if (gid in self.nodes.keys()) and (did in self.nodes.keys()):  # avoid singleton node
                    ## an evil idea: sort the raw links by the node degree from small to large
                    writer_test.writerow([i, gid, did, score])
                    continue
            if gid not in self.nodes.keys():
                self.nodes[gid] = self.add_node(gid, 'Gene')
            if did not in self.nodes.keys():
                self.nodes[did] = self.add_node(did, 'Disease')

            state =self.construct_link(gid, did, score, 'hetero')
            if state:
                writer_train.writerow([i, gid, did, score])
            if i % 10000 == 0:
                print('build_g_d_net %d '%i)
        csvfile_test.close()
        csvfile_train.close()


    def build_g_g_net(self, ggn_file,new_ggn_file):
        '''
        build gene to gene sub net
        :param ggn_file: gene-gene association file
        :return:
        '''
        df_new_ggn = pd.DataFrame(columns=('gid1', 'gid2', 'score'))
        df_new_ggn.to_csv(new_ggn_file)
        csvfile_ggn = open(new_ggn_file, 'a',newline ='')
        writer_ggn = csv.writer(csvfile_ggn)
        df_ggn = pd.read_csv(ggn_file)
        for i in range(0, len(df_ggn)):  # construct g-g network
            gid1 = str(df_ggn.loc[i, 'gene1'])
            gid2 = str(df_ggn.loc[i, 'gene2'])
            score = float(df_ggn.loc[i, 'score'])
            if self.construct_link(gid1, gid2, score, 'homo'):
                writer_ggn.writerow([i, gid1, gid2, score])
            if i % 10000 == 0:
                print('build_g_g_net %d '%i)

    def build_d_d_net(self, dds_file, d_similarity_threshold,save_file):
        '''
         build disease to disease sub net
        :param dds_file: disease-disease similarity file
        :param d_similarity_threshold:
        :return:
        '''
        df_new_dds = pd.DataFrame(columns=('did1', 'did2', 'score'))
        df_new_dds.to_csv(save_file)
        csvfile_dds = open(save_file, 'a',newline ='')
        writer_dds = csv.writer(csvfile_dds)

        df_dds = pd.read_csv(dds_file)
        for i in range(0, len(df_dds)):  # construct d-d network
            mid1 = str(df_dds.loc[i, 'UID1'])
            mid2 = str(df_dds.loc[i, 'UID2'])
            score = float(df_dds.loc[i, 'score'])
            if score < d_similarity_threshold:
                continue
            # build link between the diseaseIds in the sames meshId,the similarity between the diseaseIds is 1
            if mid1 in self.disease_map.keys():
                did_list = list(self.disease_map[mid1])
                for x in range(0, len(did_list) - 1):
                    for y in range(x + 1, len(did_list)):
                        d1, d2 = did_list[x], did_list[y]
                        if self.construct_link(d1, d2, 1.0, 'homo'):
                            writer_dds.writerow([i, d1, d2, 1.0])
            if mid2 in self.disease_map.keys():
                did_list = list(self.disease_map[mid2])
                for x in range(0, len(did_list) - 1):
                    for y in range(x + 1, len(did_list)):
                        d1, d2 = did_list[x], did_list[y]
                        if self.construct_link(d1, d2, 1.0, 'homo'):
                            writer_dds.writerow([i, d1, d2, 1.0])
            # build link between the diseaseIds in the different meshId,
            # the similarity between the diseaseIds is the score
            if (mid1 in self.disease_map.keys()) and (mid2 in self.disease_map.keys()):
                for d1 in self.disease_map[mid1]:
                    for d2 in self.disease_map[mid2]:
                        if self.construct_link(d1, d2, score, 'homo'):
                            writer_dds.writerow([i, d1, d2, score])

            if i % 10000 == 0:
                print('build_d_d_net %d '%i)

    def construct_link(self, id1, id2, score, type):
        # add bi-directional links between id1 and id2
        # add successfully, return True; else return False
        '''
        construct link between node id1 and node id2
        :param id1: node id
        :param id2: node id
        :param score: link weight
        :param type: link type :homo or hetero
        :return:
        '''
        score = 1.0
        if (id1 in self.nodes.keys()) and (id2 in self.nodes.keys()):
            state1 = self.nodes[id1].set_link(self.nodes[id2], score, type)
            state2 = self.nodes[id2].set_link(self.nodes[id1], score, type)

            if state1 and state2:
                self.links_count = self.links_count + 1
                return True
        return False

    def build_graph(self, gda_file, ggn_file, dds_file,train_file,test_file,graph_dds_file,graph_ggn_file,d_similarity_threshold=0.1,):
        '''
        build total net, include gd sub net,dd sub net,gg sub net
        '''
        self.build_g_d_net(gda_file, train_file, test_file)
        self.build_d_d_net(dds_file, d_similarity_threshold,graph_dds_file)
        self.build_g_g_net(ggn_file,graph_ggn_file)


    def random_walk(self, walk_per_vertex, walk_length, walk_file):
        '''
        deep walk to generate node sequence"
        :param walk_per_vertex: the number of walks from each vertex
        :param walk_length: walks length
        :param walk_file: save file
        '''
        self.unifrom_graph()
        print('this graph have %d node and %d links' % (self.nodes_count, self.links_count))
        print('then start random walk')
        f_walk = open(walk_file, 'w')
        for node in self.nodes.values():
            for w in range(0, walk_per_vertex):
                current = node
                text = current.node_id
                for i in range(0, walk_length - 1):
                    current = current.random_next()
                    text = text + '\t' + current.node_id
                text = text + '\n'
                f_walk.write(text)
                # f_walk.write('\n')
        f_walk.close()

def main(files_home):
    starttime = datetime.now()
    print('start build graph at ', starttime)
    gda_file = os.path.join(files_home, files_name['gda_file'])
    dds_file = os.path.join(files_home, files_name['dds_file'])
    ggn_file = os.path.join(files_home, files_name['ggn_file'])
    dmap_file = os.path.join(files_home, files_name['dmap_file'])
    walk_file = os.path.join(files_home, files_name['walk_file'])
    test_file = os.path.join(files_home, files_name['test_file'])
    train_file = os.path.join(files_home, files_name['train_file'])
    word2vec = os.path.join(files_home, files_name['word2vec'])
    graph_dds_file = os.path.join(files_home, files_name['graph_dds_file'])
    graph_ggn_file = os.path.join(files_home, files_name['graph_ggn_file'])
    g = Graph()
    g.load_disease_map(dmap_file)
    g.build_graph(gda_file, ggn_file, dds_file, train_file, test_file,graph_dds_file,graph_ggn_file)
    g.random_walk(40, 40, walk_file)
    # word2vec model
    skip_gram(walk_file, word2vec)
    endtime = datetime.now()
    print('build graph finish ! run spend ', endtime - starttime, ' s')

if __name__ == '__main__':
    files_home = files_name['file_home']
    main(files_home)






