import numpy as np
import torch.nn as nn
import torch.optim as optim

from mytest.env import EnvArgs, create_MTLEnv
from mytest.net import MTN, STNS
from mytest.datautil import read_data_from_file, MTDataset, MTDataset_Split
from mytest.testutil import test_net
from mytest.train import train_base, train_RMTL, train_RLPolicy
from mytest.arg import PPOArgs, EnvArgs, TrainArgs


def reward_fn(info):
    if not info.get('losses'):
        return 0
    array = np.asanyarray(info['losses'])
    half = array.shape[-1] // 2
    start_losses = array[:, :half].mean(axis=-1)
    end_losses = array[:, half:].mean(axis=-1)
    return start_losses.std() - end_losses.std()


def state_fn(info):
    if not info.get('losses'):
        return np.zeros(info['num_task'])
    end_losses = np.asanyarray(info['losses'])[:, -1]
    min = end_losses.min()
    max = end_losses.max()
    end_losses = (end_losses - min) / (max - min)
    return end_losses


def test_RMTL():
    policy = train_RLPolicy(args, TrainArgs, PPOArgs)
    env_net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    env = create_MTLEnv(env_net, args, trainbatcher, reward_fn, state_fn)
    net = train_RMTL(env, policy, 400)
    return net

def test_STL():
    net = STNS(feature_dim, args.hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    train_base(net, trainbatcher, criterion, optimizer, 200)
    return net

def test_MTL():
    net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    train_base(net, trainbatcher, criterion, optimizer, 200)
    return net


if __name__ == '__main__':
    args = EnvArgs(reward_fn, state_fn)

    data, label, task_interval, num_task, num_class = read_data_from_file(args.data_path)
    feature_dim = data.shape[-1]
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

    model = test_RMTL()
    # model = test_STL()
    # model = test_MTL()

    error = test_net(model, testdata, testlabel, test_task_interval)
    print(error)
