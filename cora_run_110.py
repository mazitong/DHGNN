
import torch
import numpy as np
import copy
from time import time
import torch.optim as optim
from models import DHGCF
from utils import parser
from utils import loaddata_link
from operator import itemgetter
import random as rd
import multiprocessing
import heapq
from utils import metrics
import collections
import itertools
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score


def test(model, users_to_test, num,args,test_set,neg_test_dic):

    neg_dict= neg_test_dic
    if True:
        u_g_embeddings= model( drop_flag=False)
        rate=model.rating(u_g_embeddings,u_g_embeddings)


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

def train_func(model,args,mask):
    t0 = time()

    """
    *********************************************************
    Train.
    """
    stopping_step = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)

    best_auc, best_ap = 0, 0
    test_neg_dict, _ = loaddata_link.sample_neg(test_dic, edge_dic, nums)
    users_to_test = list(test_dic.keys())  # test中的uid，全部的user



    for epoch in range(args.epoch):
        model.train()
        t1 = time()
        epoch_loss, mf_loss, emb_loss = 0., 0., 0.

        u_g_embeddings = model(drop_flag=args.node_dropout_flag)
        rate_batch = torch.matmul(u_g_embeddings, u_g_embeddings.t())

        train_neg_dict, u_i_matrix = loaddata_link.sample_neg(train_dic, edge_dic, nums)
        neg_mask = u_i_matrix == -1

        pred_p = torch.mul(mask, rate_batch)
        pred_n = torch.mul(neg_mask ,rate_batch)

        def flaten(a):
            a = a.flatten()
            nonzero = torch.nonzero(a)
            b = torch.index_select(a, dim=0, index=nonzero.squeeze())
            return b

        pred_p = flaten(pred_p)
        pred_n = flaten(pred_n)

        # pred_n calculate 0
        if pred_n.shape<pred_p.shape:
            pred_n=torch.cat([pred_n,torch.zeros(pred_p.size(0)-pred_n.size(0))],dim=0)

        # 1 if an edge exists, else 0
        label = torch.cat((torch.ones(n_train), torch.zeros(n_train)), dim=0)

        pred = (torch.cat((pred_p, pred_n), dim=0))

        decay = eval(args.regs)[0]
        maxi = torch.nn.LogSigmoid()(pred_p - pred_n)
        mf_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(u_g_embeddings) ** 2) / 2
        emb_loss = decay * regularizer / nums

        epoch_loss = mf_loss + emb_loss

        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        schedular.step()

        train_auc = roc_auc_score(label, torch.sigmoid(pred).detach().numpy())
        train_ap = average_precision_score(label, torch.sigmoid(pred).detach().numpy())

        # validation
        model.eval()



        val_auc, val_ap = test(model, users_to_test, nums, args, test_dic, test_neg_dict)

        if best_auc < val_auc:
            best_auc = val_auc
            best_ap = val_ap
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            perf_str = 'Epoch %d [%.1fs]: loss==[%.5f=%.5f + %.5f],auc=[%.5f],ap=[%.5f]' % (
                epoch, time() - t1, epoch_loss, mf_loss, emb_loss, train_auc, train_ap)
            print(perf_str)
            print('current: val_auc=[%.5f], val_ap=[%.5f])' % (val_auc, val_ap))
            print('best: val_auc=[%.5f], val_ap=[%.5f]' % (best_auc, best_ap))

        best_auc, stopping_step, should_stop = early_stopping(val_auc, best_auc,
                                                              stopping_step, expected_order='acc', flag_step=200)

        # *********************************************************
        # # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
    t2 = time()


    model.load_state_dict(best_model_wts)


    test_auc, test_ap = test(model, users_to_test, nums, args, test_dic, test_neg_dict)

    print('training time: ', t2 - t0)
    print('test: test_auc=[%.5f], test_ap=[%.5f])' % (test_auc, test_ap))
    #print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
    t3 = time()
    return best_auc,best_ap

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.dataset='cora'
    args.alpha=0.6
    changestr=str(args.alpha)
    best=[]
    for i in range(10):
        args.split='/split'+str(i)
        print('-------------------------------------------------------')
        print('start! dataset:'+args.dataset+'  '+args.split)
        L, nums, edge_dic, train_dic, test_dic, neg_test_dic, n_train, n_test, user_item_matrix,tml= loaddata_link.readdata1(
            args)

        user_item_matrix = torch.tensor((user_item_matrix))
        mask = user_item_matrix == 1



        model = DHGCF(nums, L, args).to(args.device)
        best_auc,best_ac=train_func(model, args, mask)
        best.append([best_auc,best_ac])
    j=0
    for i in best:
        print('split',j,':',i)
        j+=1
    best=np.array(best)
    np.savetxt('./data/compare/'+args.dataset+ changestr +'_result1.txt', best[:,0], fmt='%.5f')
    np.savetxt('./data/compare/' + args.dataset + changestr + '_result2.txt', best[:, 1], fmt='%.5f')







