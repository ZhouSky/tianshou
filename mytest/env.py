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
    def __init__(self, env_net: nn.Module, optimizer: optim, criterion, data_batcher, iter_step: int, reward_fn,
                 state_fn, max_iter=0, max_epoch=0, discrete=True, device=torch.device('cpu')):
        super(MTLEnv, self).__init__()
        assert (max_iter > 0 and max_epoch == 0) or (max_iter == 0 and max_epoch > 0)

        self.env_net = env_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_batcher = data_batcher
        self.discrete = discrete
        self.device = device

        self.iter_step = iter_step
        self.max_iter_epoch = np.ceil(
            data_batcher.data.shape[0] / (data_batcher.batch_size * data_batcher.num_task * data_batcher.num_class)
        ).astype(np.int32)
        self.max_iter = max_iter if max_iter > 0 else max_epoch * self.max_iter_epoch
        # self.max_iter -= self.max_iter % self.iter_step
        self.max_epoch = np.ceil(self.max_iter / self.max_iter_epoch)

        self.reward_fn = reward_fn
        self.state_fn = state_fn

        self.coe_clip = (0.01, 100)
        if self.discrete:
            self.coe_mul = (0.99, 1.0, 1.01)

        self.observation_space = Box(shape=(env_net.num_task,), low=0, high=1)
        self.action_space = Discrete(len(self.coe_mul) ** env_net.num_task) if self.discrete else Box(
            shape=(env_net.num_task,), low=self.coe_clip[0], high=self.coe_clip[1])

        self.seed()
        self.reset()

    def seed(self, seed=0):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.iter = 0
        self.info = {'num_task': self.env_net.num_task}
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

        if self.discrete:
            self.info['action'] = [self.coe_mul[(action % 3 ** (i + 1)) // (3 ** i)] for i in
                                   range(self.info['num_task'])]
        else:
            self.info['action'] = action

        coes = []
        for i in range(self.info['num_task']):
            if self.discrete:
                coes.append(max(self.coe_clip[0], min(self.coe_clip[1], self.coes[i] * self.info['action'][i])))
            else:
                coes.append(action[i])
            self.coes[i] = coes[i]
            # self.coes[i] *= self.info['action'][i]
            # self.coes[i] = max(self.coe_clip[0], min(self.coe_clip[1], self.coes[i]))

        self.info['coes'] = coes
        self.info['losses'] = [[] for _ in range(self.env_net.num_task)]
        self.info['iter'] = []
        self.info['epoch'] = []
        base = self.env_net.num_class * self.data_batcher.batch_size
        for i in range(self.iter_step):
            sampled_data, sampled_label, _, _ = self.data_batcher.get_next_batch()
            sampled_data = torch.from_numpy(sampled_data).to(self.device)
            sampled_label = torch.from_numpy(np.asarray(sampled_label == 1).nonzero()[-1]).to(self.device)

            losses = []
            for t in range(self.env_net.num_task):
                output = self.env_net(sampled_data[t * base: (t + 1) * base, :, ], t)
                loss = self.criterion(output, sampled_label[t * base: (t + 1) * base])
                losses.append(loss)
                self.info['losses'][t].append(loss.item())

            obj = 0
            for c, l in zip(coes, losses):
                obj += c * l
            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            self.iter += 1
            self.info['iter'].append(self.iter)
            self.info['epoch'].append(self.iter // self.max_iter_epoch)
            if self.iter >= self.max_iter:
                self.done = True
                break

        return self._get_state(), self._get_reward(), self.done, self.info


def create_MTLEnv(databatcher, args, reward_fn, state_fn, discrete=True, device=torch.device('cpu')):
    env_net = MTN(databatcher.data_dim, args.hidden_dim, databatcher.num_class, databatcher.num_task).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(env_net.parameters(), args.lr)
    env = MTLEnv(env_net, optimizer, criterion, databatcher, args.iter_step, reward_fn, state_fn,
                 max_iter=args.max_iter, max_epoch=args.max_epoch, discrete=discrete, device=device)
    return env


if __name__ == '__main__':
    args = EnvArgs()
    data, label, task_interval, num_task, num_class = read_data_from_file(args.data_path)
    databatcher = MTDataset(data, label, task_interval, num_class, args.size_task_class)
    env = create_MTLEnv(databatcher, args, lambda: None, lambda: None)
