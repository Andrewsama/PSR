import os
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim

import world


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']  # 1e-4
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        # loss, reg_loss = self.model.bpr_loss(users, pos, neg)  # 返回损失：loss, E0^2
        '''修改'''
        loss, reg_loss, attr_loss = self.model.bpr_loss(users, pos, neg)  # 返回损失：loss, E0^2
        attr_loss = attr_loss * 5.0
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        loss = loss + attr_loss

        self.opt.zero_grad()  # 优化器梯度清零
        loss.backward()       # 计算梯度
        self.opt.step()       # 优化器反向传播

        return loss.cpu().item()  # 返回损失值


def UniformSample_original(users, dataset):
    total_start = time()
    user_num = dataset.trainDataSize   # 训练集样本数
    # 从所有样本中随机采样，得到训练集样本大小相同的user
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos  # 训练集  所有用户的list [[用户0的items],[用户1的items],...]
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:   # 若再训练集中，某user没有好友，则不使用该user
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]    # 随机获得训练集中user的一个正样本
        while True:
            negitem = np.random.randint(0, dataset.m_items) # 随机获得训练集中user的一个负样本
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    if world.model_name == 'bpr':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name in ['LightGCN', 'PSR']:
        file = f"{world.model_name}-{world.dataset}-{world.config['layer']}layer-" \
               f"{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices) # 随机打乱顺序

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1) # 求取每个user正确预测的item数(若多于20则取20)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):   # 遍历当前batch中的user i
        groundTrue = test_data[i]     # 获得user i的真实items
        predictTopK = pred_data[i]    # 获得user i的预测前20个items
        pred = list(map(lambda x: x in groundTrue, predictTopK)) # 获得属于真实items的预测items列表
        pred = np.array(pred).astype("float") # 将其转为float类型的numpy数组
        r.append(pred)
    return np.array(r).astype('float')# 返回当前batch中所有user的预测正确的item list

# ====================end Metrics=============================
# =========================================================
