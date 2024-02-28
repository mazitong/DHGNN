

import torch
import numpy as np
from time import time
import torch.optim as optim
from utils import hypergraph_utils as hgut
from models import HGNN
from utils import parser
from utils import loaddata_link
from utils import baseline_utils as base_util
from operator import itemgetter
import random as rd
import multiprocessing
import heapq
from utils import metrics
import collections
import itertools
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
from models import DiGCN


def test(model, users_to_test, num,args,test_set,neg_test_dic,fts=[]):

    neg_dict= neg_test_dic
    if True:
        u_g_embeddings= model(edge_index,edge_weight ,fts)

        rate = torch.matmul(u_g_embeddings, u_g_embeddings.t())

        rate_batch = torch.sigmoid(rate)
        pos_n=0

        pred_n = []
        pred_p = []
        for u in users_to_test:
            pos_items=test_set[u]
            neg_items=neg_dict[u]
            pos_n+=len(pos_items)
            for p,n in zip(pos_items,neg_items):
                pred_p.append(rate_batch[u][p].detach().cpu().numpy())
                pred_n.append(rate_batch[u][n].detach().cpu().numpy())

        # for u, p, n in zip(users_to_test, pos_items, neg_items):
        #     pred_p.append(rate_batch[u][p])
        #     pred_n.append(rate_batch[u][n])

        label = [1] * pos_n + [0] * pos_n



        pred = pred_p + pred_n

        auc=roc_auc_score(label,pred)
        ap=average_precision_score(label,pred)

    return auc,ap

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def train_func(fts=[]):

    t0 = time()
    stopping_step = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)
    # schedular = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=int(2000),
    #     gamma=float(args.gamma)
    # )

    best_auc, best_ap = 0, 0
    lossdic=[]
    times=0
    for epoch in range(args.epoch):
        model.train()
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        BCEloss = torch.nn.BCELoss()

        u_g_embeddings = model(edge_index,edge_weight,fts)
        rate_batch = torch.matmul(u_g_embeddings, u_g_embeddings.t())

        pred_p1 = mask * rate_batch

        train_neg_dict, u_i_matrix = loaddata_link.sample_neg(train_dic, edge_dic, nums)

        neg_mask = u_i_matrix == -1
        neg_dict = train_neg_dict

        pred_n1 = neg_mask * rate_batch

        def flaten(a):
            a = a.flatten()
            nonzero = torch.nonzero(a)
            b = torch.index_select(a, dim=0, index=nonzero.squeeze())
            return b

        pred_p1 = flaten(pred_p1)
        pred_n1 = flaten(pred_n1)

        # pred_p1=pred_p1[torch.randperm(pred_p1.size(0))]

        if pred_n1.shape < pred_p1.shape:
            pred_n1 = torch.cat([pred_n1, torch.zeros(pred_p1.size(0) - pred_n1.size(0))], dim=0)
        # rate_batch = torch.sigmoid(rate)
        pred_n = torch.zeros(n_train)
        pred_p = torch.zeros(n_train)

        # lens = range(len(train_dic.keys()))
        i = 0
        for u in train_dic.keys():

            pos_i = train_dic[u]

            neg_i = neg_dict[u]
            for p, n in zip(pos_i, neg_i):
                pred_p[i] = rate_batch[u][p]
                pred_n[i] = rate_batch[u][n]
                i += 1

        label = torch.cat((torch.ones(n_train), torch.zeros(n_train)), dim=0)

        pred = (torch.cat((pred_p, pred_n), dim=0))
        decay = eval(args.regs)[0]
        maxi = torch.nn.LogSigmoid()(pred_p1 - pred_n1)

        mf_loss = -1 * torch.mean(maxi)
        posloss = -torch.log(torch.sigmoid(pred_p1) + decay).mean()
        negloss = -torch.log(1 - torch.sigmoid(pred_n1) + decay).mean()

        regularizer = (torch.norm(u_g_embeddings) ** 2) / 2
        emb_loss = decay * regularizer / nums
        # mf_loss=BCEloss(torch.sigmoid(pred),label)
        batch_loss = mf_loss
        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()
        schedular.step()

        train_auc = roc_auc_score(label, torch.sigmoid(pred).detach().numpy())
        train_ap = average_precision_score(label, torch.sigmoid(pred).detach().numpy())

        loss += batch_loss

        users_to_test = list(test_dic.keys())  # test中的uid，全部的user

        model.eval()
        test_auc, test_ap = test(model, users_to_test, nums, args, test_dic, test_neg_dict,fts)
        if best_auc < test_auc:
            best_auc = test_auc
            best_ap = test_ap

        if (epoch + 1) % 50 != 0:
            perf_str = 'Epoch %d [%.1fs]: loss==[%.5f=%.5f + %.5f],auc=[%.5f],ap=[%.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss, train_auc, train_ap)
            print(perf_str)
        times+=time()-t1
        if (epoch + 1) % 10 == 0:
            print('current: test_auc=[%.5f], test_ap=[%.5f])' % (test_auc, test_ap))
            print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
        best_auc, stopping_step, should_stop = early_stopping(test_auc, best_auc,
                                                              stopping_step, expected_order='acc', flag_step=200)
        lossdic.append(loss.detach().numpy())
        # *********************************************************
        # # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            print('times:', times, 'times/epoch', times / (epoch + 1))
            break
        # if epoch == 800:
        #     print('times:', times, 'times/epoch', times / (epoch + 1))
        #     break
    # np.savetxt('./data/' + args.dataset + args.split + '_DiGCN' + '_loss.txt', lossdic)
    print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
    t1 = time()
    print('training time: ',t1-t0)
    users_to_test = list(test_dic.keys())  # test中的uid，全部的user

    test_auc, test_ap = test(model, users_to_test, nums, args, test_dic,test_neg_dict)
    print('current: test_auc=[%.5f], test_ap=[%.5f])' % (test_auc, test_ap))
    print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
    t3 = time()
    return best_auc,best_ap
if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.dataset = 'dblp'
    alpha=0.1
    best = []
    for i in range(10):
        args.split = '/split' + str(i)
        print('-------------------------------------------------------')
        print('start! dataset:' + args.dataset + '  ' + args.split)
        args.alpha=0.1
        # args.dataset='amazon_computers'

        L, nums, edge_dic, train_dic, test_dic, neg_test_dic, n_train, n_test ,user_item_matrix,tm= loaddata_link.readdata(args)
        fts=[]
        # fts, L, nums, edge_dic, train_dic, test_dic, neg_test_dic, n_train, n_test, user_item_matrix = loaddata_link.readdata_with_fts(args)
        #
        # fts=torch.tensor(fts,dtype=torch.float32)
        # args.embed_size = fts.shape[1]

        user_item_matrix = torch.tensor((user_item_matrix))

        test_neg_dict, _ = loaddata_link.sample_neg(test_dic, edge_dic, nums)
        train_neg_dict, u_i_matrix = loaddata_link.sample_neg(train_dic, edge_dic, nums)
        mask = user_item_matrix > 0
        neg_mask = u_i_matrix == -1

        adj_coo=sp.coo_matrix(torch.mul(mask,user_item_matrix))

        indices = np.vstack((adj_coo.row, adj_coo.col))#把非零值的行列标号拼接为（i，j）的下标矩阵
        indices = torch.from_numpy(indices).long()

        edge_index,edge_weight=base_util.get_appr_directed_adj(alpha,indices,nums,dtype=torch.float32)





        model=DiGCN(nums,args.embed_size,args.layer_size,args.node_dropout[0])
        best_auc, best_ac = train_func(fts)
        best.append([best_auc, best_ac])
    j = 0
    bauc=[]
    bap=[]
    for i in best:
        print('split', j, ':', i)

        j += 1
    best = np.array(best)
    np.savetxt('./data/'+args.dataset+'/digcn_result.txt', best, fmt='%.5f')
    np.savetxt('./data/'+args.dataset+'/digcn_result1.txt', best[:,0], fmt='%.5f')
    np.savetxt('./data/'+args.dataset+'/digcn_result2.txt', best[:, 1], fmt='%.5f')
