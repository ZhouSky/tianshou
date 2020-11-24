import torch
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

from mytest.net import MTN
from mytest.env import create_MTLEnv, MTLEnv
from mytest.datautil import read_data_from_file, MTDataset


def train_RLPolicy(EnvArgs, TrainArgs, PPOArgs, discrete=True):
    print('-----Start train RL policy-----')
    data, label, task_interval, num_task, num_class = read_data_from_file(EnvArgs.data_path)
    feature_dim = data.shape[-1]
    databatcher = MTDataset(data, label, task_interval, num_class, EnvArgs.size_task_class)
    env_net = MTN(feature_dim, EnvArgs.hidden_dim, num_class, num_task)

    train_envs = SubprocVectorEnv(
        [lambda: create_MTLEnv(env_net, EnvArgs, databatcher, EnvArgs.reward_fn, EnvArgs.state_fn, discrete) for _ in
         range(TrainArgs.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: create_MTLEnv(env_net, EnvArgs, databatcher, EnvArgs.reward_fn, EnvArgs.state_fn, discrete) for _ in
         range(TrainArgs.test_num)])

    np.random.seed(TrainArgs.seed)
    torch.manual_seed(TrainArgs.seed)
    train_envs.seed(TrainArgs.seed)
    test_envs.seed(TrainArgs.seed)

    env = create_MTLEnv(env_net, EnvArgs, databatcher, EnvArgs.reward_fn, EnvArgs.state_fn, discrete)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    del env

    net = Net(TrainArgs.layer_num, state_shape)
    actor = Actor(net, action_shape, discrete=discrete)
    if discrete:
        dist = torch.distributions.Categorical
    else:
        dist = torch.distributions.normal.Normal

    critic = Critic(net)
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
    env.workers[0].env.max_iter = env.workers[0].env.max_iter_epoch * max_epoch
    data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={})
    data.obs = env.reset()
    done = False
    tag = 'train-MTL'
    while not done:
        action = policy(data).act
        data.obs, rew, done, info = env.step(action)
        info = info[0]
        losses = np.asarray(info['losses'])
        for ind, ite in enumerate(info['iter']):
            if ite % env.workers[0].env.max_iter_epoch == 0:
                epoch = ite // env.workers[0].env.max_iter_epoch
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
