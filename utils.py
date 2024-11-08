import os, time, multiprocessing
import gc
import random
import numpy as np
import scipy
import scipy.spatial
import torch
import math
import sys
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import torch.nn.functional as F
import scipy.sparse as sp

#CompGCN
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def ccorr(a, b):
	return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

#EAkit
def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def multi_cal_rank(task, distance, distanceT, top_k, args):
    acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
    mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
    for i in range(len(task)):
        ref = task[i]
        indices = distance[i, :].argsort()
        rank = np.where(indices == ref)[0][0]
        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for j in range(len(top_k)):
            if rank < top_k[j]:
                acc_l2r[j] += 1
    for i in range(len(task)):
        ref = task[i]
        indices = distanceT[:, i].argsort()
        rank = np.where(indices == ref)[0][0]
        mean_r2l += (rank + 1)
        mrr_r2l += 1.0 / (rank + 1)
        for j in range(len(top_k)):
            if rank < top_k[j]:
                acc_r2l[j] += 1
    del distance, distanceT
    gc.collect()
    return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)


def multi_cal_neg(pos, task, triples, r_hs_dict, r_ts_dict, ids, k):
    neg = []
    for _ in range(k):
        neg_part = []
        for idx, tas in enumerate(task):
            (h, r, t) = pos[tas]
            temp_scope, num = True, 0
            while True:
                h2, r2, t2 = h, r, t
                choice = np.random.binomial(1, 0.5)
                if choice:
                    if temp_scope:
                        h2 = random.sample(r_hs_dict[r], 1)[0]
                    else:
                        for id in ids:
                            if h2 in id:
                                h2 = random.sample(id, 1)[0]
                                # break
                else:
                    if temp_scope:
                        t2 = random.sample(r_ts_dict[r], 1)[0]
                    else:
                        for id in ids:
                            if t2 in id:
                                t2 = random.sample(id, 1)[0]
                                # break
                if (h2, r2, t2) not in triples:
                    break
                else:
                    num += 1
                    if num > 10:
                        temp_scope = False
            neg_part.append((h2, r2, t2))
        neg.append(neg_part)        
    return neg

def multi_typed_sampling(pos, triples, ills, ids, k, params, thread=10):
    t_ = time.time()
    if len(pos[0]) == 2:    # triple: 1:k
        raise NotImplementedError("typed_sampling is not supported in ills sampling")
    triples = set(triples)
    r_hs_dict, r_ts_dict = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs_dict:
            r_hs_dict[r] = set()
        if r not in r_ts_dict:
            r_ts_dict[r] = set()
        r_hs_dict[r].add(h)
        r_ts_dict[r].add(t)
    tasks = div_list(np.array(range(len(pos)), dtype=np.int32), thread)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(multi_cal_neg, (pos, task, triples, r_hs_dict, r_ts_dict, ids, k)))
    pool.close()
    pool.join()
    neg_part = [[] for _ in range(k)]
    for res in reses:
        item = res.get()    # (k, n, 3)
        for i in range(k):
            neg_part[i].extend(item[i])
    neg = []
    for part in neg_part:
        neg.extend(part)
    # print("\tmulti_typed_sampling time cost: {:.3f} s".format(time.time() - t_))
    return neg


def typed_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    if len(pos[0]) == 2:    # triple: 1:k
        raise NotImplementedError("typed_sampling is not supported in ills sampling")
    triples = set(triples)
    r_hs_dict, r_ts_dict = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs_dict:
            r_hs_dict[r] = set()
        if r not in r_ts_dict:
            r_ts_dict[r] = set()
        r_hs_dict[r].add(h)
        r_ts_dict[r].add(t)
    tasks = div_list(np.array(range(len(pos)), dtype=np.int32), 1)
    neg_part = multi_cal_neg(pos, tasks[0], triples, r_hs_dict, r_ts_dict, ids, k)
    neg = []
    for part in neg_part:
        neg.extend(part)
    # print("\ttyped_sampling time cost: {:.3f} s".format(time.time() - t_))
    return neg

def nearest_neighbor_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    emb = params["emb"]
    metric = params["metric"]
    if len(pos[0]) == 3:    # triple: 1:k
        sorted_id = [sorted(ids[0]), sorted(ids[1])]
        distance = [- sim(emb[sorted_id[0]], emb[sorted_id[0]], metric=metric, normalize=False, csls_k=0), - sim(emb[sorted_id[1]], emb[sorted_id[1]], metric=metric, normalize=False, csls_k=0)]
        cache_dict = {}
        neg = []
        triples = set(triples)
        for _ in range(k):
            for (h, r, t) in pos:
                base_h = 0 if h in ids[0] else 1
                base_t = 0 if t in ids[0] else 1
                while True:
                    h2, r2, t2 = h, r, t
                    choice = np.random.binomial(1, 0.5)
                    if choice:
                        if h not in cache_dict:
                            indices = np.argsort(distance[base_h][sorted_id[base_h].index(h), :])  # descending=False
                            cache_dict[h] = np.array(sorted_id[base_h])[indices[1 : ]].tolist()
                        h2 = random.sample(cache_dict[h][ : k], 1)[0]
                    else:
                        if t not in cache_dict:
                            indices = np.argsort(distance[base_t][sorted_id[base_t].index(t), :])  # descending=False
                            cache_dict[t] = np.array(sorted_id[base_t])[indices[1 : ]].tolist()
                        t2 = random.sample(cache_dict[t][ : k], 1)[0]
                    if (h2, r2, t2) not in triples:
                        break
                neg.append((h2, r2, t2))
    elif len(pos[0]) == 2:  # ill: 1:2k
        neg_left = []
        distance = - sim(emb[pos[:, 0]], emb[pos[:, 0]], metric=metric, normalize=False, csls_k=0)
        for idx in range(len(pos)):
            indices = np.argsort(distance[idx, :])  # descending=False
            neg_left.append(pos[:, 0][indices[1 : k+1]])
        neg_left = np.stack(neg_left, axis=1).reshape(-1, 1)
        neg_right = []
        distance = - sim(emb[pos[:, 1]], emb[pos[:, 1]], metric=metric, normalize=False, csls_k=0)
        for idx in range(len(pos)):
            indices = np.argsort(distance[idx, :])  # descending=False
            neg_right.append(pos[:, 1][indices[1 : k+1]])
        neg_right = np.stack(neg_right, axis=1).reshape(-1, 1)
        neg_left = np.concatenate((neg_left, np.tile(pos, (k, 1))[:, 1].reshape(-1, 1)), axis=1).tolist()
        neg_right = np.concatenate((np.tile(pos, (k, 1))[:, 0].reshape(-1, 1), neg_right), axis=1).tolist()
        neg = neg_left + neg_right
        del distance
        gc.collect()
    else:
        raise NotImplementedError
    # print("\tnearest_neighbor_sampling time cost: {:.3f} s".format(time.time() - t_))
    return neg


def random_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    if len(pos[0]) == 3:    # triple: 1:k
        neg = []
        triples = set(triples)
        for _ in range(k):
            for (h, r, t) in pos:
                ent_set = ids[0] if h in ids[0] else ids[1]
                while True:
                    h2, r2, t2 = h, r, t
                    choice = np.random.binomial(1, 0.5)
                    if choice:
                        h2 = random.sample(ent_set, 1)[0]
                    else:
                        t2 = random.sample(ent_set, 1)[0]
                    if (h2, r2, t2) not in triples:
                        break
                neg.append((h2, r2, t2))
    elif len(pos[0]) == 2:  # ill: 1:2k
        neg_left, neg_right = [], []
        ills = set([(e1, e2) for (e1, e2) in ills])
        for _ in range(k):
            for (e1, e2) in pos:
                e11 = random.sample(ids[0] - {e1}, 1)[0]
                neg_left.append((e11, e2))
                e22 = random.sample(ids[1] - {e2}, 1)[0]
                neg_right.append((e1, e22))
        neg = neg_left + neg_right
    else:
        raise NotImplementedError
    # print("\trandom_sampling time cost: {:.3f} s".format(time.time() - t_))
    return neg


# --- Code from AliNet (https://github.com/nju-websoft/AliNet) ---
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
        #embeds1 = embeds1 / np.maximum(np.linalg.norm(embeds1, axis=-1, keepdims=True), 1e-7)
        #embeds2 = embeds2 / np.maximum(np.linalg.norm(embeds2, axis=-1, keepdims=True), 1e-7)
        #sim_mat = np.matmul(embeds1, embeds2.T)
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        # print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat

def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)
# --- Code from AliNet(https://github.com/nju-websoft/AliNet) end ---