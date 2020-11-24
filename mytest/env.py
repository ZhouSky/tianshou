from typing import Any

import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mytest.datautil import read_data_from_file, MTDataset_Split, MTDataset
from mytest.net import MTN
from mytest.arg import EnvArgs


class MTLEnv(gym.Env):
    def __init__(self, env_net: nn.Module, optimizer: optim, criterion, data_batcher, iter_step: int, max_iter: int,
                 reward_fn, state_fn, discrete=True):
        self.env_net = env_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_batcher = data_batcher
        self.discrete = discrete

        self.iter_step = iter_step
        self.max_iter = max_iter
        self.max_iter_epoch = np.ceil(
            data_batcher.data.shape[0] / (data_batcher.batch_size * data_batcher.num_task * data_batcher.num_class)
        ).astype(np.int32)
        self.max_epoch = np.ceil(self.max_iter / self.max_iter_epoch)
        self.iter = 0
        self.done = False

        self.reward_fn = reward_fn
        self.state_fn = state_fn
        self.info = {'num_task': env_net.num_task}

        self.coes = np.ones((env_net.num_task,))
        self.coe_clip = (0.01, 100)
        if self.discrete:
            self.coe_mul = (0.99, 1.0, 1.01)

        self.observation_space = Box(shape=(env_net.num_task,), low=0, high=1)
        self.action_space = Discrete(len(self.coe_mul) ** env_net.num_task) if self.discrete else Box(
            shape=(env_net.num_task,), low=self.coe_clip[0], high=self.coe_clip[1])

        self.seed()

    def seed(self, seed=0):
        np.random.seed(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.iter = 0
        self.coes = np.ones((self.info['num_task'],))
        self.env_net.reset()
        return self._get_state()

    def _get_reward(self):
        """Generate a non-scalar reward if ma_rew is True."""
        return self.reward_fn(self.info)

    def _get_state(self):
        """Generate state(observation) of MyTestEnv"""
        return self.state_fn(self.info)

    def setMaxEpoch(self, max_epoch):
        self.max_iter = self.max_iter_epoch * max_epoch
        self.max_epoch = max_epoch

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')

        coes = []
        if self.discrete:
            self.info['action'] = [self.coe_mul[(action % 3 ** (i + 1)) // (3 ** i)] for i in range(self.info['num_task'])]
        else:
            self.info['action'] = action
        for i in range(self.info['num_task']):
            coes.append(max(self.coe_clip[0], min(self.coe_clip[1], self.coes[i] * self.info['action'][i])))
            if self.discrete:
                self.coes[i] = coes[i]
            # self.coes[i] *= self.info['action'][i]
            # self.coes[i] = max(self.coe_clip[0], min(self.coe_clip[1], self.coes[i]))

        self.info['coes'] = coes
        self.info['losses'] = [[] for _ in range(self.env_net.num_task)]
        self.info['iter'] = []
        base = self.env_net.num_class * self.data_batcher.batch_size
        for i in range(self.iter_step):
            sampled_data, sampled_label, _, _ = self.data_batcher.get_next_batch()
            sampled_data = torch.from_numpy(sampled_data)
            sampled_label = torch.from_numpy(np.asarray(sampled_label == 1).nonzero()[-1])

            losses = []
            for t in range(self.env_net.num_task):
                output = self.env_net(sampled_data[t * base: (t + 1) * base, :, ], t)
                loss = self.criterion(output, sampled_label[t * base: (t + 1) * base])
                losses.append(loss)
                self.info['losses'][t].append(loss.item())

            self.optimizer.zero_grad()
            obj = 0
            for c, l in zip(coes, losses):
                obj += c * l
            obj.backward()
            self.optimizer.step()

            self.iter += 1
            self.info['iter'].append(self.iter)
            if self.iter >= self.max_iter:
                self.done = True
                break

        return self._get_state(), self._get_reward(), self.done, self.info


def create_MTLEnv(env_net, args, databatcher, reward_fn, state_fn, discrete=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(env_net.parameters(), args.lr)
    env = MTLEnv(env_net, optimizer, criterion, databatcher, args.iter_step, args.max_iter, reward_fn, state_fn, discrete)
    return env


if __name__ == '__main__':
    args = EnvArgs()
    data, label, task_interval, num_task, num_class = read_data_from_file(args.data_path)
    feature_dim = data.shape[-1]
    databatcher = MTDataset(data, label, task_interval, num_class, args.size_task_class)
    env_net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    env = create_MTLEnv(env_net, args, databatcher, lambda: None, lambda: None)
