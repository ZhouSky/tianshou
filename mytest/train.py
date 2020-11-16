import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import pprint
import os

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, Batch
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import Net

from mytest.net import MTN, STNS
from mytest.env import create_MTLEnv, EnvArgs, MTLEnv
from mytest.testutil import test_net
from mytest.datautil import read_data_from_file, MTDataset, MTDataset_Split


class PPOArgs:
    gamma = 0.99
    max_grad_norm = 0.5
    eps_clip = 0.2
    vf_coef = 0.5
    ent_coef = 0.0


class TrainArgs:
    lr = 1e-3
    layer_num = 1
    seed = 0
    training_num = 1
    test_num = 1
    buffer_size = 256
    logdir = 'log'
    epoch = 1  # each epoch will test policy once
    step_per_epoch = 1  # num of iter per epoch for policy train
    collect_per_step = 1  # num of eps per iter for policy train
    repeat_per_collect = 2  # meaningless
    batch_size = 64  # meaningless


def train_RLPolicy(EnvArgs, TrainArgs, PPOArgs):
    print('-----Start train RL policy-----')
    data, label, task_interval, num_task, num_class = read_data_from_file(EnvArgs.data_path)
    feature_dim = data.shape[-1]
    databatcher = MTDataset(data, label, task_interval, num_class, EnvArgs.size_task_class)
    env_net = MTN(feature_dim, EnvArgs.hidden_dim, num_class, num_task)
    train_envs = create_MTLEnv(env_net, EnvArgs, databatcher, EnvArgs.reward_fn, EnvArgs.state_fn)
    test_envs = create_MTLEnv(env_net, EnvArgs, databatcher, EnvArgs.reward_fn, EnvArgs.state_fn)

    np.random.seed(TrainArgs.seed)
    torch.manual_seed(TrainArgs.seed)
    train_envs.seed(TrainArgs.seed)
    test_envs.seed(TrainArgs.seed)

    state_shape = train_envs.observation_space.shape or train_envs.observation_space.n
    action_shape = train_envs.action_space.shape or train_envs.action_space.n
    net = Net(TrainArgs.layer_num, state_shape)
    actor = Actor(net, action_shape)
    critic = Critic(net)
    optimizer_rl = torch.optim.Adam(set(list(
        actor.parameters()) + list(critic.parameters())), lr=TrainArgs.lr)
    dist = torch.distributions.Categorical

    policy = PPOPolicy(
        actor, critic, optimizer_rl, dist, PPOArgs.gamma,
        max_grad_norm=PPOArgs.max_grad_norm,
        eps_clip=PPOArgs.eps_clip,
        vf_coef=PPOArgs.vf_coef,
        ent_coef=PPOArgs.ent_coef,
        action_range=None)

    train_collector = Collector(
        policy, train_envs, ReplayBuffer(TrainArgs.buffer_size),
        preprocess_fn=None)
    test_collector = Collector(policy, test_envs, preprocess_fn=None)

    result = onpolicy_trainer(
        policy, train_collector, test_collector, TrainArgs.epoch,
        TrainArgs.step_per_epoch, TrainArgs.collect_per_step, TrainArgs.repeat_per_collect,
        TrainArgs.test_num, TrainArgs.batch_size)
    pprint.pprint(result)
    print('-----End train RL policy-----')

    return policy


def train_RMTL(env, policy, max_epoch):
    print('-----Start train MTL net-----')
    if isinstance(env, MTLEnv):
        env = DummyVectorEnv([lambda: env])
    env.workers[0].env.max_iter = env.workers[0].env.max_iter_epoch * max_epoch
    data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={})
    data.obs = env.reset()
    done = False
    while not done:
        action = policy(data).act
        data.obs, rew, done, info = env.step(action)
        losses = np.asarray(info[0]['losses'])
        for ind, ite in enumerate(info[0]['iter']):
            if ite % env.workers[0].env.max_iter_epoch == 0:
                print('Epoch %d, Iter: %d, Reward: %f, loss: %s' % (
                    ite // env.workers[0].env.max_iter_epoch, ite, rew, losses[:, ind]))
    print('-----End train MTL net-----')
    return env.workers[0].env.env_net


def train_base(model, databatcher, criterion, optimizer, max_epoch):
    max_iter_epoch = np.ceil(
        databatcher.data.shape[0] / (databatcher.batch_size * databatcher.num_task * databatcher.num_class)
    ).astype(np.int32)
    base = databatcher.num_class * databatcher.batch_size
    for iter in range(max_iter_epoch * max_epoch):
        sampled_data, sampled_label, _, _ = databatcher.get_next_batch()
        num_epoch = iter // max_iter_epoch
        outputs = []
        for t in range(databatcher.num_task):
            output = model(torch.from_numpy(sampled_data[t * base: (t + 1) * base, :]), t)
            outputs.append(output)
        output = torch.cat(outputs, 0)

        sampled_label = torch.tensor(
            [np.where(sampled_label[_] == 1)[0][0] for _ in range(sampled_label.shape[0])]
        )

        loss = criterion(output, sampled_label) * databatcher.num_task
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % max_iter_epoch == 0:
            print("Epoch %d, Iter %d, training loss %g" % (num_epoch, iter, loss))
