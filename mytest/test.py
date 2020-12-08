import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

from mytest.env import create_MTLEnv
from mytest.net import MTN, STNS
from mytest.datautil import read_data_from_file, MTDataset, MTDataset_Split
from mytest.testutil import test_net
from mytest.train import train_base, train_RMTL, train_RLPolicy
from mytest.arg import PPOArgs, EnvArgs, TrainArgs


# def reward_fn(info):
#     if not info.get('losses'):
#         return 0
#     array = np.asanyarray(info['losses'])
#     half = array.shape[-1] // 2
#     start_losses = array[:, :half].mean(axis=-1)
#     end_losses = array[:, half:].mean(axis=-1)
#     return start_losses.std() - end_losses.std()

# def reward_fn(info):
#     if not info.get('losses'):
#         return 0
#     array = np.asanyarray(info['losses'])
#     return 1 / np.exp(array.mean(axis=-1).max())

def reward_fn(info):
    if not info.get('losses'):
        return 0
    array = np.asanyarray(info['losses'])
    mid = array.shape[-1] // 2
    left_losses = array[:, :mid].mean(axis=-1)
    right_losses = array[:, mid:].mean(axis=-1)
    return 1 / (array.mean(axis=-1).std() + np.exp(array.mean(axis=-1).max())) + (left_losses - right_losses).sum()


def state_fn(info):
    if not info.get('losses'):
        return np.zeros(info['num_task'])
    avg_losses = np.asanyarray(info['losses']).mean(axis=-1)
    min = avg_losses.min()
    max = avg_losses.max()
    end_losses = (avg_losses - min) / (max - min)
    return end_losses


def test_RMTL(databatcher, args, device=torch.device('cpu')):
    discrete = True
    policy = train_RLPolicy(args, TrainArgs, PPOArgs, discrete, device)
    env = create_MTLEnv(databatcher, args, reward_fn, state_fn, discrete, device)
    writer = SummaryWriter(os.path.join(TrainArgs.logdir, 'MTL', 'mtl'))
    net = train_RMTL(env, policy, 500, writer)
    return net


def test_STL(trainbatcher, args, device=torch.device('cpu')):
    net = STNS(trainbatcher.data_dim, args.hidden_dim, trainbatcher.num_class, trainbatcher.num_task).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    writer = SummaryWriter(os.path.join(TrainArgs.logdir, 'STL-Base'))
    train_base(net, trainbatcher, criterion, optimizer, 1000, writer, device)
    return net


def test_MTL(trainbatcher, args, device=torch.device('cpu')):
    net = MTN(trainbatcher.data_dim, args.hidden_dim, trainbatcher.num_class, trainbatcher.num_task).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), args.lr)
    writer = SummaryWriter(os.path.join(TrainArgs.logdir, 'MTL-Base'))
    train_base(net, trainbatcher, criterion, optimizer, 2000, writer, device)
    return net


def main():
    args = EnvArgs(reward_fn, state_fn)
    gpu = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if gpu else 'cpu')
    print(device)

    data, label, task_interval, num_task, num_class = read_data_from_file(args.data_path)
    data_split = MTDataset_Split(data, label, task_interval, num_class)
    (
        traindata,
        trainlabel,
        train_task_interval,
        testdata,
        testlabel,
        test_task_interval,
    ) = data_split.split(args.train_size)
    trainbatcher = MTDataset(
        traindata, trainlabel, train_task_interval, num_class, args.size_task_class
    )

    model = test_RMTL(trainbatcher, args, device)
    # model = test_STL(trainbatcher, args, device)
    # model = test_MTL(trainbatcher, args, device)

    test_train = test_net(model, traindata, trainlabel, train_task_interval, device)
    test_test = test_net(model, testdata, testlabel, test_task_interval, device)
    return test_train, test_test


if __name__ == '__main__':
    mp.set_start_method('spawn')
    start = time.time()
    print(main())
    print(time.time() - start)
