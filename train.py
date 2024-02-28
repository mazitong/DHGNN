
import torch
import numpy as np
from time import time
import torch.optim as optim
from models import DHGCF1
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

        from sklearn.metrics import roc_auc_score,average_precision_score
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


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fts, L, nums, edge_dic, train_dic, test_dic, neg_test_dic, n_train, n_test ,user_item_matrix= loaddata_link.readdata_with_fts(args)
    fts=torch.tensor(fts,dtype=torch.float)
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    # initializer = torch.nn.init.xavier_uniform_
    # embedding_dict = initializer(torch.empty(nums,
    #                                          args.embed_size))
    model = DHGCF1(fts,nums, L, args).to(args.device)
    user_item_matrix = torch.tensor((user_item_matrix))
    test_neg_dict,_ = loaddata_link.sample_neg(test_dic, edge_dic, nums)
    train_neg_dict,u_item_matrix = loaddata_link.sample_neg(train_dic, edge_dic, nums)


    mask=user_item_matrix>0
    neg_mask= u_item_matrix==-1
    t0 = time()
    """
    *********************************************************
    Train.
    """
    stopping_step = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0003)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100, 600],
                                               gamma=0.9)


    best_auc, best_ap = 0, 0

    for epoch in range(args.epoch):
        model.train()
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        BCEloss=torch.nn.BCELoss()
        schedular.step()



        u_g_embeddings = model(drop_flag=args.node_dropout_flag)
        rate_batch = torch.matmul(u_g_embeddings, u_g_embeddings.t())

        pred_p1=mask*rate_batch

        train_neg_dict, u_i_matrix = loaddata_link.sample_neg(train_dic, edge_dic, nums)

        neg_mask = u_i_matrix == -1
        neg_dict = train_neg_dict

        pred_n1=neg_mask*rate_batch
        # print(neg_mask[0][neg_dict[0][1]])
        # print(neg_dict[0])
        # print(pred_n1[0][neg_dict[0][1]])

        def flaten(a):
            a=a.flatten()
            nonzero = torch.nonzero(a)
            b = torch.index_select(a,dim=0,index=nonzero.squeeze())
            return b



        pred_p1=flaten(pred_p1)
        pred_n1=flaten(pred_n1)

        #pred_p1=pred_p1[torch.randperm(pred_p1.size(0))]



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

        pred = (torch.cat((pred_p1, pred_n1), dim=0))
        decay = eval(args.regs)[0]
        maxi = torch.nn.LogSigmoid()(pred_p1 - pred_n1)

        mf_loss = -1 * torch.mean(maxi)
        posloss=-torch.log(torch.sigmoid(pred_p1)+decay).mean()
        negloss=-torch.log(1-torch.sigmoid(pred_n1)+decay).mean()

        regularizer = (torch.norm(u_g_embeddings) ** 2) / 2

        #mf_loss=BCEloss(torch.sigmoid(pred),label)
        batch_loss=mf_loss
        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()


        from sklearn.metrics import roc_auc_score, average_precision_score

        # print(len(label), len(pred))

        # batch_loss=model.create_loss(u_g_embeddings,pos_i_g_embeddings,data_generator.mask,data_generator.R)
        # batch_mf_loss, batch_emb_loss=0,0
        # batch_loss=torch.nn.CrossEntropyLoss()(model.rating(u_g_embeddings,i_g_embeddings),torch.tensor(data_generator.R[users,:]) )


        train_auc = roc_auc_score(label, torch.sigmoid(pred).detach().numpy())
        train_ap = average_precision_score(label, torch.sigmoid(pred).detach().numpy())

        loss += batch_loss

        users_to_test = list(test_dic.keys())  # test中的uid，全部的user

        model.eval()
        test_auc, test_ap = test(model, users_to_test, nums, args, test_dic, test_neg_dict)
        if best_auc < test_auc:
            best_auc = test_auc
            best_ap = test_ap

        if (epoch + 1) % 50 != 0:
            perf_str = 'Epoch %d [%.1fs]: loss==[%.5f=%.5f + %.5f],auc=[%.5f],ap=[%.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss, train_auc, train_ap)
            print(perf_str)
        if (epoch + 1) % 10 == 0:
            print('current: test_auc=[%.5f], test_ap=[%.5f])' % (test_auc, test_ap))
            print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
        best_auc, stopping_step, should_stop = early_stopping(test_auc, best_auc,
                                                              stopping_step, expected_order='acc', flag_step=200)

        # *********************************************************
        # # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
    t2 = time()
    users_to_test = list(test_dic.keys())  # test中的uid，全部的user
    print(test_dic)
    test_auc, test_ap = test(model, users_to_test, nums, args, test_dic, test_neg_dict)
    print('current: test_auc=[%.5f], test_ap=[%.5f])' % (test_auc, test_ap))
    print('best: best_auc=[%.5f], best_ap=[%.5f]' % (best_auc, best_ap))
    t3 = time()

