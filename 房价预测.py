import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

all_features = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index # 取出所有的数值特征

# 减去均值，除以方差
all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean())
                                                                / (x.std()))
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.fillna(all_features.mean())

num_train = train.shape[0]

train_features = all_features[:num_train].as_matrix().astype(np.float32)
test_features = all_features[num_train:].as_matrix().astype(np.float32)
train_labels = train.SalePrice.as_matrix()[:, None].astype(np.float32)
test_labels = test.SalePrice.as_matrix()[:, None].astype(np.float32)

from torch import nn

def get_model():
    # todo: 使用 nn.Sequential 来构造多层神经网络，注意第一层的输入
    model = nn.Sequential(nn.Linear(331,32),
                          nn.ReLU(),
                          nn.Linear(32,1314)
                        )
    return model

# 可以调整的超参

batch_size = 32
epochs = 100
use_gpu = False
lr = 1
weight_decay = 10

criterion = nn.MSELoss()# todo: 使用 mse 作为 loss 函数

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from utils import get_rmse_log

# todo: 将所有的 feature 和 label 都转成 torch 的 Tensor
train_features = torch.from_numpy(train_features)
test_features = torch.from_numpy(test_features)
train_labels = torch.from_numpy(train_labels)
test_labels = torch.from_numpy(test_labels)

# 构建一个数据的迭代器
def get_data(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=4)


def train_model(model, x_train, y_train, x_valid, y_valid, epochs, lr, weight_decay):
    metric_log = dict()
    metric_log['train_loss'] = list()
    if x_valid is not None:
        metric_log['valid_loss'] = list()

    train_data = get_data(x_train, y_train, batch_size, True)
    if x_valid is not None:
        valid_data = get_data(x_valid, y_valid, batch_size, False)
    else:
        valid_data = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))  # todo: 构建优化器，推荐使用 Adam，也可以尝试一下别的优化器

    for e in range(epochs):
        # 训练模型
        running_loss = 0
        model.train()
        for data in train_data:
            x, y = data
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            x = Variable(x)
            y = Variable(y)

            # todo: 前向传播
            y_out = model(x)

            print(y_out.shape)

            # todo: 计算 loss
            loss = criterion(y_out, y)
            # todo: 反向传播，更新参数
            loss.backward()
            optimizer.step()

        metric_log['train_loss'].append(get_rmse_log(model, x_train, y_train, use_gpu))

        # 测试模型
        if x_valid is not None:
            metric_log['valid_loss'].append(get_rmse_log(model, x_valid, y_valid, use_gpu))
            print_str = 'epoch: {}, train loss: {:.3f}, valid loss: {:.3f}' \
                .format(e + 1, metric_log['train_loss'][-1], metric_log['valid_loss'][-1])
        else:
            print_str = 'epoch: {}, train loss: {:.3f}'.format(e + 1, metric_log['train_loss'][-1])
        if (e + 1) % 10 == 0:
            print(print_str)
            print()

    # =======不要修改这里的内容========
    # 可视化
    figsize = (10, 5)
    fig = plt.figure(figsize=figsize)
    plt.plot(metric_log['train_loss'], color='red', label='train')
    if valid_data is not None:
        plt.plot(metric_log['valid_loss'], color='blue', label='valid')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


# train_data = get_data(train_features, train_labels, batch_size, True)
model = get_model()
# for data in train_data:
#
#     x, y = data
#     if use_gpu:
#         x = x.cuda()
#         y = y.cuda()
#     x = Variable(x)
#     y = Variable(y)
#     y_out = model(x)

train_model(model, train_features, train_labels, test_features, test_labels, epochs, lr, weight_decay)
