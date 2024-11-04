import time
import numpy as np
import scipy.sparse as sp
from utils import *

class AlignmentData:

    def __init__(self, data_dir="data/DBP15K/zh_en", train_rate=0.3, val_rate=0.0, fold = 1):
        t_ = time.time()

        self.labeled_alignment = set()
        self.boot_triple_idx = []
        self.boot_pair_dix = []

        self.train_rate = train_rate
        self.val_rate = val_rate
        # 读取 ent_ids_
        # {实体名称:实体ID} {实体ID:实体名称} [图1的所有实体ID, 图2的所有实体ID]
        self.ent2id_dict, self.id2ent_dict, [self.kg1_ent_ids, self.kg2_ent_ids] = self.load_dict(data_dir + "/ent_ids_", file_num=2)
        # with open("ent_name", "w", encoding="utf-8") as f:
        #     for key, value in sorted(self.id2ent_dict.items()):
        #         f.write(str(key) +' ' + str(value) + '\n')
        # 读取 rel_ids_
        # {关系名称:关系ID} {关系ID:关系名称} [图1的所有关系ID, 图2的所有关系ID]
        self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids] = self.load_dict(data_dir + "/rel_ids_", file_num=2)

        # 总实体数
        self.ent_num = len(self.ent2id_dict)
        # 总关系数
        self.rel_num = len(self.rel2id_dict)

        # 读取 三元组triples_ 和 预对齐ill_ent_ids,都是元组
        self.triple_idx = self.load_triples(data_dir + "/triples_", file_num=2)
        
        self.ill_idx = self.load_triples(data_dir + "/ill_ent_ids", file_num=1)

        # 把预对齐按 train_rate, val_rate 分为 train+val+test
        np.random.shuffle(self.ill_idx)
        self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = \
            np.array(self.ill_idx[:int(len(self.ill_idx) // 1 * train_rate)], dtype=np.int32), \
            np.array(self.ill_idx[int(len(self.ill_idx) // 1 * train_rate) : int(len(self.ill_idx) // 1 * (train_rate+val_rate))], dtype=np.int32), \
            np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (train_rate+val_rate)):], dtype=np.int32)
        """

        # 加入另外的数据集
        if not("_V1" in data_dir or "_V2" in data_dir):
            self.ill_idx = self.load_triples(data_dir + "/ill_ent_ids", file_num=1)
            np.random.shuffle(self.ill_idx)
            self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = np.array(self.ill_idx[:int(len(self.ill_idx) // 1 * train_rate)], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * train_rate) : int(len(self.ill_idx) // 1 * (train_rate+val_rate))], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (train_rate+val_rate)):], dtype=np.int32)
        else:
            self.ill_train_idx = self.load_triples(data_dir + "/721_5fold/" + str(fold) + "/train.txt", file_num=1)
            self.ill_val_idx = self.load_triples(data_dir + "/721_5fold/" + str(fold) + "/valid.txt", file_num=1)
            self.ill_test_idx = self.load_triples(data_dir + "/721_5fold/" + str(fold) + "/test.txt", file_num=1)
            self.ill_idx = self.ill_train_idx + self.ill_val_idx + self.ill_test_idx
            self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = np.array(self.ill_train_idx, dtype = np.int32), np.array(self.ill_val_idx, dtype = np.int32), np.array(self.ill_test_idx, dtype = np.int32)
        """






        # with open("ref_pairs", "w") as f:
        #     for v in self.ill_test_idx:
        #         f.write(str(v[0]) + '\t' + str(v[1]) + '\n')
        # with open("sup_pairs", "w") as f:
        #     for v in self.ill_train_idx:
        #         f.write(str(v[0]) + '\t' + str(v[1]) + '\n')
        # assert 0

        # 获取稀疏图,没有去除自环,没有加入反向边,没有交换ill
        # [[head, tail], ...] [[1], ...]             [[r], ...]
        self.sparse_edges_idx, self.sparse_values, self.sparse_rels_idx = self.gen_sparse_graph_from_triples()
        
        # assert (share != swap or (share == False and swap == False))
        # if share:
        #     self.triple_idx = self.share(self.triple_idx, self.ill_train_idx)   # 1 -> 2:base
        #     self.kg1_ins_ids = (self.kg1_ins_ids - set(self.ill_train_idx[:, 0])) | set(self.ill_train_idx[:, 1])
        #     self.ill_train_idx = []
        # if swap:
        #     self.triple_idx = self.swap(self.triple_idx, self.ill_train_idx)



        self.init_time = time.time() - t_

    def load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t"))) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def load_dict(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = {}, {}, []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids

    # def gen_sparse_graph_from_triples(self):
    #     edge_dict = {}
    #     #print(self.triple_idx[0], self.triple_idx[1])
    #     for (h, r, t) in self.triple_idx:
    #         if (h, t) not in edge_dict:
    #             edge_dict[(h, t)] = []
    #         edge_dict[(h, t)].append(r)
    #     edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
    #     values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
    #     r_ij = [r for (h, t) in edge_dict for r in edge_dict[(h, t)]]
    #     edges = np.array(edges, dtype=np.int32)
    #     values = np.array(values, dtype=np.float32)
    #     r_ij = np.array(r_ij, dtype=np.int32)
    #     #print(len(edges), len(values), len(r_ij), len(self.triple_idx))
    #     return edges, values, r_ij

    def gen_sparse_graph_from_triples(self):
        edge_dict = {}
        tmp_triple_idx = []
        for (h, r, t) in self.triple_idx:
            if h != t:
                if (h, t) not in edge_dict:
                    edge_dict[(h, t)] = []
                    edge_dict[(t, h)] = []
                edge_dict[(h, t)].append(r)

                # 添加反向边
                edge_dict[(t, h)].append(-r)
                #edge_dict[(t, h)].append(r + self.rel_num)
                #tmp_triple_idx.append(tuple([h, r + self.rel_num, t]))
        #self.rel_num *= 2

        #self.triple_idx.extend(tmp_triple_idx)

        edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
        values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
        r_ij = [abs(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]

        # 添加自环
        for i in range(self.ent_num):
            edges.append([i, i])
            values.append(1)
            r_ij.append(self.rel_num) 
            #self.triple_idx.append(tuple([i, self.rel_num, i]))
        self.rel_num += 1

        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)
        r_ij = np.array(r_ij, dtype=np.int32)
        return edges, values, r_ij       


    def share(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        new_triples = []
        for (h, r, t) in triples:
            if h in from_1_to_2_dict:
                h = from_1_to_2_dict[h]
            if t in from_1_to_2_dict:
                t = from_1_to_2_dict[t]
            new_triples.append((h, r, t))
        new_triples = list(set(new_triples))
        return new_triples
    
    def swap(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        from_2_to_1_dict = dict(ill[:, ::-1])
        new_triples = []
        for (h, r, t) in triples:
            new_triples.append((h, r, t))
            if h in from_1_to_2_dict:
                new_triples.append((from_1_to_2_dict[h], r, t))
            if t in from_1_to_2_dict:
                new_triples.append((h, r, from_1_to_2_dict[t]))
            if h in from_2_to_1_dict:
                new_triples.append((from_2_to_1_dict[h], r, t))
            if t in from_2_to_1_dict:
                new_triples.append((h, r, from_2_to_1_dict[t]))
        new_triples = list(set(new_triples))
        return new_triples


    def __repr__(self): # print 时调用
        return self.__class__.__name__ + " dataset summary:" + \
            "\n\tent_num: " + str(self.ent_num) + \
            "\n\trel_num: " + str(self.rel_num) + \
            "\n\ttriple_idx: " + str(len(self.triple_idx)) + \
            "\n\ttrain_rate: " + str(self.train_rate) + "\tval_rate: " + str(self.val_rate) + \
            "\n\till_idx(train/test/val): " + str(len(self.ill_idx)) + " = " + str(len(self.ill_train_idx)) + " + " + str(len(self.ill_test_idx)) + " + " + str(len(self.ill_val_idx)) + \
            "\n\tsprase_edges_idx: " + str(len(self.sparse_edges_idx)) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


if __name__ == '__main__':
    
    # TEST
    d = AlignmentData()
    print(d)
    #print(d.triple_idx[0], d.sparse_edges_idx[0], d.sparse_rels_idx[0]) (14688, 590, 16302) [14688 16302] 590.0