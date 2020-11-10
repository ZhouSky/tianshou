import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import pprint
import os
import argparse

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import Net

from mytest.env import create_env


class EnvArgs:
    lr = 1e-3
    size_task_class = 4  # means batch size: s_t_c * n_t * n_c
    hidden_dim = [600]
    train_size = 0.8
    once_iter = 20
    max_iter = 1000
    data_path = "./Office_Caltech_alexnet.txt"


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
    buffer_size = 200
    logdir = 'log'
    epoch = 2
    step_per_epoch = 8
    collect_per_step = 1
    repeat_per_collect = 2
    batch_size = 64


def reward_fn(reward_info):
    if not reward_info.get('losses'):
        return 0
    start_losses = np.array([reward_info['losses'][t][0] for t in range(reward_info['num_task'])])
    end_losses = np.array([reward_info['losses'][t][-1] for t in range(reward_info['num_task'])])
    return start_losses.std() - end_losses.std()


def state_fn(state_info):
    if not state_info.get('losses'):
        return np.zeros(state_info['num_task'])
    end_losses = np.array([state_info['losses'][t][-1] for t in range(state_info['num_task'])])
    end_losses = (end_losses - end_losses.mean()) / end_losses.std()
    return end_losses


def learn_controller(EnvArgs, TrainArgs, PPOArgs):
    print('Start train controller!')
    env = create_env(EnvArgs, reward_fn, state_fn)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    train_envs = create_env(EnvArgs, reward_fn, state_fn)
    # train_envs = DummyVectorEnv([
    #     lambda: create_env(EnvArgs, reward_fn, state_fn)
    #     for _ in range(TrainArgs.training_num)])
    test_envs = create_env(EnvArgs, reward_fn, state_fn)
    # test_envs = DummyVectorEnv([
    #     lambda: create_env(EnvArgs, reward_fn, state_fn)
    #     for _ in range(TrainArgs.test_num)])

    np.random.seed(TrainArgs.seed)
    torch.manual_seed(TrainArgs.seed)
    train_envs.seed(TrainArgs.seed)
    test_envs.seed(TrainArgs.seed)

    net = Net(TrainArgs.layer_num, state_shape)
    # print(action_shape)
    actor = Actor(net, action_shape)
    critic = Critic(net)
    optimizer_rl = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=TrainArgs.lr)
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

    writer = SummaryWriter(os.path.join(TrainArgs.logdir, 'MTL', 'ppo'))

    def stop_fn(mean_rewards):
        # if env.env.spec.reward_threshold:
        #     return mean_rewards >= env.spec.reward_threshold
        # else:
        return False

    result = onpolicy_trainer(
        policy, train_collector, test_collector, TrainArgs.epoch,
        TrainArgs.step_per_epoch, TrainArgs.collect_per_step, TrainArgs.repeat_per_collect,
        TrainArgs.test_num, TrainArgs.batch_size, stop_fn=stop_fn, writer=writer)
    pprint.pprint(result)
    print('End train controller!')

    return policy


if __name__ == '__main__':

    policy = learn_controller(EnvArgs, TrainArgs, PPOArgs)
    # Let's watch its performance!
    env = create_env(EnvArgs, reward_fn, state_fn)
    collector = Collector(policy, env, preprocess_fn=None)
    result = collector.collect(n_step=2000, render=None)
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')

