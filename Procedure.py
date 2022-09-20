import numpy as np
import torch

import utils
import world

# 按batch_size将所有users通过计算正负样本的bpr损失，都做一次反向传播，并返回损失值
def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(allusers, dataset) # [user,pos,neg], [times list]
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)   # 打乱顺序
    total_batch = len(users) // world.config['bpr_batch_size'] + 1  # 按batch_size分成多个batch
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,  # 创建一个yield迭代器
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        # 梯度下降位置
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)  # 反向传播，并返回损失值
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):   # 输入的是一个batch的数据
    sorted_items = X[0].numpy()  # 得到当前batch中用户的预测前20个item 列表
    groundTrue = X[1]            # 得到当前batch中用户测试集中的item 列表
    r = utils.getLabel(groundTrue, sorted_items) # 得到当前batch中用户预测正确的item 列表
    pre, recall, ndcg = [], [], []
    for k in world.topks:   # [10, 20]
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)  # 返回precision和recall的值
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = world.config['test_u_batch_size']    # 100
    # dict: 测试集    key为用户，value为items {用户i:[items]}
    if cold:
        testDict: dict = dataset.coldTestDict
    else:
        testDict: dict = dataset.testDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)  # 20
    results = {'precision': np.zeros(len(world.topks)),#2
               'recall': np.zeros(len(world.topks)),   #2
               'ndcg': np.zeros(len(world.topks))}     #2
    with torch.no_grad():
        users = list(testDict.keys())  # 获得测试数据中的users
        try:
            assert u_batch_size <= len(users) / 10  # 若测试中的users数大于1000，则报错
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1  # 迭代次数
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):  # 按batch_size输入数据
            allPos = dataset.getUserPosItems(batch_users)      # 获得用户的训练集中的正item
            groundTrue = [testDict[u] for u in batch_users]    # 获得用户的测试集中的正item
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device) # 将数据变为long类型后输入到gpu中

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 获得测试集user和所有items的分数
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))   # 值为item对应的user下标
                exclude_items.extend(items)                    # 将batch中user的正item拼接成一维数组
            # 避免取值取到训练集中的正item
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)   # 默认返回每一行的前k大的值和下标
            del rating
            # 二维列表，第一维为batch，第二维为user
            users_list.append(batch_users)
            # 三维列表，第一维为batch，第二维为user，第三维为user预测的前k相似item的下标
            rating_list.append(rating_K.cpu())     # 预测前20
            # 三维列表，第一维为batch，第二维为user，第三维为user测试集中相连的item的下标
            groundTrue_list.append(groundTrue)     # 真实相连
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []   # dict 返回各指标的值
        for x in X:  # 左： rating_list[i]      右：groundTrue_list[i]
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
        return results
