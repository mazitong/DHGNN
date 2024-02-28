import os
import time
import copy

import numpy as np
import torch
import torch.optim as optim
import pprint as pp
import utils1.hypergraph_utils as hgut
from utils1.hypergraph_utils import _edge_dict_to_H,construct_H_with_KNN,hyperedge_concat,construct_H_with_KNN_cosin
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
import utils1.generate_dihyg as gd
from collections import Counter


os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


# DH_D=np.load("DDD-H.npy",allow_pickle=True)
# print("DH_H:",DH_D.shape)
# print("DH_H:",DH_D[0])
def print_label_dist(label_col):
    c = Counter(label_col)
    print(f'label is {c}')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')#读取config文件
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# initialize data
#P,fts,lbls,edge_dict,idx_train,idx_test=gd.construct_dihg('cora')
# fts,lbls,H_head,H_tail,edge_dict,idx_train,idx_test=gd.read_data('citeseer')
# G=gd.generate_G_from_H(fts.shape[0],H_head,H_tail)
# G=[]

'''三个地方要改。最后记录best_val
1. 数据集：cora,citeseer
2.模型名：44、45行
3. G：53行
'''
fts,lbls,G,idx_train,idx_test=gd.load_data('cora','0')#十个分割：0-9
n_class = int(lbls.max()) + 1#类别数



idx_val=idx_test[0:500]
print_label_dist(lbls[idx_train])
print('lbls:',lbls)
print_label_dist(lbls)

print("fts:",fts.shape)#特征向量矩阵 n×m维


fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)

idx_val=torch.Tensor(idx_val).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=20, print_freq=500):
    since = time.time()#计算训练时间
    best_model_wts = copy.deepcopy(model.state_dict())#存放模型效果最好时的参数
    best_acc = 0.0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:#打印
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')
        #每一个epoch都进行训练和 验证？测试
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()#更新学习率，当epoch=milestones才执行更新
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            #得到下标
            idx = idx_train if phase == 'train' else idx_val

            # Iterate over data.把梯度置零
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)#调用HGNN中的forword函数训练
                loss = criterion(outputs[idx], lbls[idx])#计算损失（输出，标签）
                # 输出每行最大值_,以及最大值的下标preds
                #训练得到的outputs维度为结点数×class，输出每行最大的列的下标即为类别
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                # 反向传播，修正w和b
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)#损失
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])#准确率

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            # epoch_acc = accuracy(outputs[idx],lbls[idx])
            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model 当准确率更高时，修改模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:.4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')


    # load best model weights
    # 将最优结果的权重载入相应模型
    model.load_state_dict(best_model_wts)
    model.eval()
    optimizer.zero_grad()
    outputs = model(fts, G)
    acc=0
    _, preds = torch.max(outputs, 1)
    acc += torch.sum(preds[idx_test] == lbls.data[idx_test])  # 准确率
    epoch_acc = acc.double() / len(idx_test)
    print('testacc:',epoch_acc)
    return model

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features,G), test_labels)


def _main():


    # HGNN模型，构造HGNN类
    model_ft = HGNN(in_ch=fts.shape[1],#特征维度
                    n_class=n_class,
                    n_hid=cfg['n_hid'],#隐藏层维度
                    dropout=cfg['drop_out'])

    model_ft = model_ft.to(device)
    print(model_ft.parameters())
    #优化器 :待优化参数的iterable,学习率,权重衰减（L2惩罚）
    optimizer = optim.Adam(model_ft.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    #动态调整学习率 gamma为每一次调整倍数
    # 一旦达到某一阶段(milestones：epoch索引列表)时，就可以通过gamma系数降低每个参数组的学习率
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    #交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    #开始训练： 模型，损失函数，优化器，学习率调整，最大epoch，打印轮次
    model_ft = train_model(model_ft, criterion, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq'])
    # test_acc=test_regression(model_ft, fts, lbls[idx_val])
    # print('test_acc:',test_acc)



if __name__ == '__main__':
    _main()
