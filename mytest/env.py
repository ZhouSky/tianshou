from typing import Any

import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mytest.datautil import read_data_from_file, MTDataset_Split, MTDataset


class MTN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_task):
        super(MTN, self).__init__()
        shared_models = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
        for i in range(1, len(hidden_dim)):
            shared_models.extend(
                [nn.Linear(hidden_dim[i - 1], hidden_dim[i]), nn.ReLU()]
            )
        self.shared_models = nn.Sequential(*shared_models)
        self.task_models = nn.ModuleList(
            [nn.Linear(hidden_dim[-1], output_dim) for _ in range(num_task)]
        )
        self.num_task = num_task
        self.num_class = output_dim
        self.reset()

    def forward(self, x, task_index):
        # print(x[0][:10])
        out = self.shared_models(x)
        # print(out[0][:10])
        out = self.task_models[task_index](out)
        # print(out[0])
        # print(F.softmax(out, dim=-1)[0])
        return out

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)


class MTLEnv(gym.Env):
    def __init__(self, env_net, optimizer, criterion, data_batcher, once_iter, max_iter, reward_fn, state_fn):
        self.env_net = env_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_batcher = data_batcher

        self.once_iter = once_iter
        self.max_iter = max_iter
        self.iter = 0
        self.done = False
        self.coes = np.ones((4,))
        self.coe_low = 0.01
        self.coe_high = 100
        self.coe_mul = [0.99, 1.0, 1.01]
        # self.num_task = env_net.num_task

        self.reward_fn = reward_fn
        self.reward_info = {'num_task':env_net.num_task}
        self.state_fn = state_fn
        self.state_info = {'num_task':env_net.num_task}

        self.observation_space = Box(shape=(env_net.num_task,), low=0, high=1)
        self.action_space = Discrete(3**4)

        self.seed()

    def seed(self, seed=0):
        np.random.seed(seed)
        return [seed]

    def reset(self, state=0):
        self.done = False
        self.iter = 0
        self.coes = np.ones((4,))
        self.env_net.reset()
        return self._get_state()

    def _get_reward(self):
        """Generate a non-scalar reward if ma_rew is True."""
        if not self.reward_info:
            return 0
        return self.reward_fn(self.reward_info)

    def _get_state(self):
        """Generate state(observation) of MyTestEnv"""
        if not self.state_info:
            return np.zeros(self.env_net.num_task)
        return self.state_fn(self.state_info)

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')
        # print(action)
        for i in range(4):
            self.coes[i] *= self.coe_mul[(action % 3**(i + 1)) // (3**i)]
        # print(self.iter, self.coes)
        print_freq = 0
        if print_freq != 0:
            print_freq = self.once_iter // print_freq

        all_losses = [[] for _ in range(self.env_net.num_task)]
        base = self.env_net.num_class * self.data_batcher.batch_size
        for i in range(self.once_iter):
            sampled_data, sampled_label, _, _ = self.data_batcher.get_next_batch()
            # print(sampled_data[0, 0])
            sampled_label = torch.tensor(
                [np.where(sampled_label[_] == 1)[0][0] for _ in range(sampled_label.shape[0])]
            )
            losses = []
            for t in range(self.env_net.num_task):
                output = self.env_net(
                    torch.from_numpy(sampled_data[t * base: (t + 1) * base, :, ]), t,
                )
                # for o, l in zip(output, sampled_label[t * base: (t + 1) * base]):
                #     print(o.argmax().item(), l.item())
                    # assert o.argmax().item() == l.item()
                # if t == 1:
                #     print(output)
                loss = self.criterion(
                    output, sampled_label[t * base: (t + 1) * base],
                )
                # if t == 0:
                #     print(sampled_data[t * base: (t + 1) * base, :, ][0][:5], output[0], sampled_label[t * base: (t + 1) * base][0], loss)
                losses.append(loss)
                all_losses[t].append(loss.item())
            # print(losses)
            self.optimizer.zero_grad()
            obj = 0
            for c, l in zip(self.coes, losses):
                obj += c * l
            obj.backward()
            self.optimizer.step()

            self.iter += 1
            if print_freq != 0 and (i + 1) % print_freq == 0:
                print(
                    "iter %d, training loss %s, objective value %g" % (self.iter, [i.item() for i in losses], obj)
                )
                ...

            if self.iter >= self.max_iter:
                self.done = True
                # assert False
                break

        self.reward_info['losses'] = all_losses
        self.state_info['losses'] = all_losses
        return self._get_state(), self._get_reward(), self.done, {}


def create_env(args, reward_fn, state_fn):
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
    env_net = MTN(feature_dim, args.hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(env_net.parameters(), args.lr)
    env = MTLEnv(env_net, optimizer, criterion, trainbatcher, args.once_iter, args.max_iter, reward_fn, state_fn)
    return env


if __name__ == '__main__':
    lr = 1e-3
    size_task_class = 4  # means batch size: s_t_c * n_t * n_c
    hidden_dim = [600]
    train_size = 0.8
    once_iter = 20
    max_iter = 1000
    data_path = "./Office_Caltech_alexnet.txt"

    env = create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter)