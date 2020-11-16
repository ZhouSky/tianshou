import numpy as np
import torch.nn as nn
import torch.optim as optim

from mytest.env import EnvArgs, create_MTLEnv
from mytest.net import MTN, STNS
from mytest.datautil import read_data_from_file, MTDataset, MTDataset_Split
from mytest.testutil import test_net
from mytest.train import train_base, train_RMTL, train_RLPolicy, TrainArgs, PPOArgs


def reward_fn(info):
    if not info.get('losses'):
        return 0
    start_losses = np.array([info['losses'][t][0] for t in range(info['num_task'])])
    end_losses = np.array([info['losses'][t][-1] for t in range(info['num_task'])])
    return start_losses.std() - end_losses.std()


def state_fn(info):
    if not info.get('losses'):
        return np.zeros(info['num_task'])
    end_losses = np.array([info['losses'][t][-1] for t in range(info['num_task'])])
    end_losses = (end_losses - end_losses.mean()) / (end_losses.std() + 1e-8)
    return end_losses


def test_RMTL():
    policy = train_RLPolicy(EnvArgs(reward_fn, state_fn), TrainArgs, PPOArgs)
    env_net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    env = create_MTLEnv(env_net, args, trainbatcher, reward_fn, state_fn)
    net = train_RMTL(env, policy, 100)
    return net

def test_STL():
    net = STNS(feature_dim, args.hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    train_base(net, trainbatcher, criterion, optimizer, 100)
    return net

def test_MTL():
    net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    train_base(net, trainbatcher, criterion, optimizer, 100)
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

    # model = test_RMTL()
    # model = test_STL()
    model = test_MTL()

    error = test_net(model, testdata, testlabel, test_task_interval)
    print(error)
