from typing import Any

import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mytest.datautil import read_data_from_file, MTDataset_Split, MTDataset
from mytest.net import MTN


class EnvArgs:
    def __init__(self, reward_fn, state_fn):
        self.lr = 1e-3
        self.size_task_class = 4  # means batch size: s_t_c * n_t * n_c
        self.hidden_dim = [600]
        self.train_size = 0.8
        self.iter_step = 10
        self.max_iter = 500
        self.data_path = "./Office_Caltech_alexnet.txt"
        self.reward_fn = reward_fn
        self.state_fn = state_fn


class MTLEnv(gym.Env):
    def __init__(self, env_net: nn.Module, optimizer: optim, criterion, data_batcher, iter_step: int, max_iter: int, reward_fn, state_fn):
        self.env_net = env_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_batcher = data_batcher

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
        self.coe_mul = (0.99, 1.0, 1.01)

        self.observation_space = Box(shape=(env_net.num_task,), low=0, high=1)
        self.action_space = Discrete(len(self.coe_mul)**env_net.num_task)

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

        for i in range(self.info['num_task']):
            self.coes[i] *= self.coe_mul[(action % 3**(i + 1)) // (3**i)]
            self.coes[i] = max(self.coe_clip[0], min(self.coe_clip[1], self.coes[i]))

        all_losses = [[] for _ in range(self.env_net.num_task)]
        base = self.env_net.num_class * self.data_batcher.batch_size
        for i in range(self.iter_step):
            sampled_data, sampled_label, _, _ = self.data_batcher.get_next_batch()
            sampled_label = torch.tensor(
                [np.where(sampled_label[_] == 1)[0][0] for _ in range(sampled_label.shape[0])]
            )

            losses = []
            for t in range(self.env_net.num_task):
                output = self.env_net(
                    torch.from_numpy(sampled_data[t * base: (t + 1) * base, :, ]), t,
                )
                loss = self.criterion(
                    output, sampled_label[t * base: (t + 1) * base],
                )
                losses.append(loss)
                all_losses[t].append(loss.item())

            self.optimizer.zero_grad()
            obj = 0
            for c, l in zip(self.coes, losses):
                obj += c * l
            obj.backward()
            self.optimizer.step()

            self.iter += 1
            if self.iter >= self.max_iter:
                self.done = True
                break

        self.info['losses'] = all_losses
        self.info['iter'] = list(range(self.iter - self.iter_step + 1, self.iter + 1))

        return self._get_state(), self._get_reward(), self.done, self.info


def create_MTLEnv(env_net, args, databatcher, reward_fn, state_fn):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(env_net.parameters(), args.lr)
    env = MTLEnv(env_net, optimizer, criterion, databatcher, args.iter_step, args.max_iter, reward_fn, state_fn)
    return env


if __name__ == '__main__':
    args = EnvArgs()
    data, label, task_interval, num_task, num_class = read_data_from_file(args.data_path)
    feature_dim = data.shape[-1]
    databatcher = MTDataset(data, label, task_interval, num_class, args.size_task_class)
    env_net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    env = create_MTLEnv(env_net, args, databatcher, lambda: None, lambda: None)
