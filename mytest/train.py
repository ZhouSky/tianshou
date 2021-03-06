import torch
import torch.nn as nn
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

from mytest.net import MTN, EDN, STNS
from mytest.env import create_MTLEnv, MTLEnv
from mytest.datautil import read_data_from_file, MTDataset


def train_RLPolicy(EnvArgs, TrainArgs, PPOArgs, discrete=True, device=torch.device('cpu')):
    print('-----Start train RL policy-----')
    data, label, task_interval, num_task, num_class = read_data_from_file(EnvArgs.data_path)
    databatcher = MTDataset(data, label, task_interval, num_class, EnvArgs.size_task_class)

    train_envs = SubprocVectorEnv(
        [lambda: create_MTLEnv(databatcher.copy(), EnvArgs, EnvArgs.reward_fn, EnvArgs.state_fn, discrete, device) for _ in
         range(TrainArgs.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: create_MTLEnv(databatcher.copy(), EnvArgs, EnvArgs.reward_fn, EnvArgs.state_fn, discrete, device) for _ in
         range(TrainArgs.test_num)])

    np.random.seed(TrainArgs.seed)
    torch.manual_seed(TrainArgs.seed)
    train_envs.seed(TrainArgs.seed)
    test_envs.seed(TrainArgs.seed)

    env = create_MTLEnv(databatcher, EnvArgs, EnvArgs.reward_fn, EnvArgs.state_fn, discrete, device)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    del env

    net = Net(TrainArgs.layer_num, state_shape, device=device)
    # actor_post = nn.Linear()
    actor = Actor(net, action_shape, discrete=discrete).to(device)
    if discrete:
        dist = torch.distributions.Categorical
    else:
        dist = torch.distributions.normal.Normal
    critic = Critic(net).to(device)
    optimizer_rl = torch.optim.Adam(set(list(
        actor.parameters()) + list(critic.parameters())), lr=TrainArgs.lr)

    policy = PPOPolicy(
        actor, critic, optimizer_rl, dist, PPOArgs.gamma,
        max_grad_norm=PPOArgs.max_grad_norm,
        eps_clip=PPOArgs.eps_clip,
        vf_coef=PPOArgs.vf_coef,
        ent_coef=PPOArgs.ent_coef,
        action_range=None)

    train_collector = Collector(policy, train_envs, ReplayBuffer(TrainArgs.buffer_size))
    test_collector = Collector(policy, test_envs, ReplayBuffer(TrainArgs.buffer_size))

    writer = SummaryWriter(os.path.join(TrainArgs.logdir, 'MTL', 'ppo'))

    result = onpolicy_trainer(
        policy, train_collector, test_collector, TrainArgs.epoch,
        TrainArgs.step_per_epoch, TrainArgs.collect_per_step, TrainArgs.repeat_per_collect,
        TrainArgs.test_num, TrainArgs.batch_size, writer=writer)
    pprint.pprint(result)
    print('-----End train RL policy-----')

    return policy


def train_RMTL(env, policy, max_epoch, writer):
    print('-----Start train MTL net-----')
    if isinstance(env, MTLEnv):
        env = DummyVectorEnv([lambda: env])
        inner_env = env.workers[0].env
    inner_env.setMaxEpoch(max_epoch)
    data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={})
    data.obs = env.reset()
    done = False
    tag = 'train-RMTL'
    while not done:
        action = policy(data).act
        data.obs, rew, done, info = env.step(action)
        info = info[0]
        losses = np.asarray(info['losses'])
        for ind, ite in enumerate(info['iter']):
            if ite % inner_env.max_iter_epoch == 0:
                epoch = ite // inner_env.max_iter_epoch
                # print('Epoch %d, Iter: %d, Reward: %f, loss: %s' % (
                #     epoch, ite, rew, losses[:, ind]))
                loss = {}
                action = {}
                coes = {}
                for t in range(len(losses[:, ind])):
                    loss[f'task{t}'] = losses[:, ind][t]
                    action[f'task{t}'] = info['action'][t]
                    coes[f'task{t}'] = info['coes'][t]
                writer.add_scalars(tag + '/loss', loss, epoch)
                writer.add_scalars(tag + '/action', action, epoch)
                writer.add_scalars(tag + '/coes', coes, epoch)
                writer.add_scalar(tag + '/reward', rew, epoch)
    print('-----End train MTL net-----')
    return inner_env.env_net


def train_base(model, databatcher, criterion, optimizer, max_epoch, writer=None, device=torch.device('cpu')):
    base = databatcher.num_class * databatcher.batch_size
    max_iter_epoch = np.ceil(databatcher.data.shape[0] / (base * databatcher.num_task)).astype(np.int32)
    tag = ''
    for iter in range(max_iter_epoch * max_epoch):
        sampled_data, sampled_label, _, _ = databatcher.get_next_batch()
        sampled_data = torch.from_numpy(sampled_data).to(device)
        sampled_label = torch.from_numpy(np.asarray(sampled_label == 1).nonzero()[-1]).to(device)

        # outputs = []
        losses = []
        for t in range(databatcher.num_task):
            output = model(sampled_data[t * base: (t + 1) * base, :], t)
            loss = criterion(output, sampled_label[t * base: (t + 1) * base])
            losses.append(loss)
            # outputs.append(output)
        # output = torch.cat(outputs, 0)
        loss = 0
        for l in losses:
            loss += l

        # loss = criterion(output, sampled_label) #* databatcher.num_task
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_epoch = iter // max_iter_epoch
        if iter % max_iter_epoch == 0:
            print("Epoch %d, Iter %d, training loss %g" % (num_epoch, iter, loss))
            loss = {}
            for i, l in enumerate(losses):
                loss[f'task{i}'] = l
            writer.add_scalars('loss', loss, num_epoch)
