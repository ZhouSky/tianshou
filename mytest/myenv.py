import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pprint

from datautil import read_data_from_file, MTDataset_Split, MTDataset

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import Net

from torch.utils.tensorboard import SummaryWriter


class MyEnv(gym.Env):

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
        print_freq = 20
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
            if (i + 1) % print_freq == 0:
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
    # end_losses = (end_losses - end_losses.mean()) / (end_losses.std() + 1e-8)
    return end_losses


def create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter):
    data, label, task_interval, num_task, num_class = read_data_from_file(data_path)
    feature_dim = data.shape[-1]
    data_split = MTDataset_Split(data, label, task_interval, num_class)
    (
        traindata,
        trainlabel,
        train_task_interval,
        testdata,
        testlabel,
        test_task_interval,
    ) = data_split.split(train_size)
    trainbatcher = MTDataset(
        traindata, trainlabel, train_task_interval, num_class, size_task_class
    )
    env_net = MTN(feature_dim, hidden_dim, num_class, num_task)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(env_net.parameters(), lr)
    env = MyEnv(env_net, optimizer, criterion, trainbatcher, once_iter, max_iter, reward_fn, state_fn)
    return env


lr = 1e-3
size_task_class = 4  # means batch size: s_t_c * n_t * n_c
hidden_dim = [600]
train_size = 0.8
once_iter = 20
max_iter = 1000
data_path = "./Office_Caltech_alexnet.txt"

seed = 0
training_num = 1
test_num = 1
layer_num = 1
gamma = 0.99
max_grad_norm = 0.5
eps_clip = 0.2
vf_coef = 0.5
ent_coef = 0.0
buffer_size = 200
logdir = 'log'
epoch = 3
step_per_epoch = 8
collect_per_step = 4
repeat_per_collect = 2
batch_size = 64


env = create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
# train_envs = gym.make(task)
train_envs = DummyVectorEnv([
    lambda: create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter)
    for _ in range(training_num)])
# test_envs = gym.make(task)
test_envs = DummyVectorEnv([
    lambda: create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter)
    for _ in range(test_num)])
# seed
np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed)
# model
net = Net(layer_num, state_shape)
actor = Actor(net, action_shape)
critic = Critic(net)
optimizer_rl = torch.optim.Adam(list(
    actor.parameters()) + list(critic.parameters()), lr=lr)
dist = torch.distributions.Categorical
policy = PPOPolicy(
    actor, critic, optimizer_rl, dist, gamma,
    max_grad_norm=max_grad_norm,
    eps_clip=eps_clip,
    vf_coef=vf_coef,
    ent_coef=ent_coef,
    action_range=None)
# collector
train_collector = Collector(
    policy, train_envs, ReplayBuffer(buffer_size),
    preprocess_fn=None)
test_collector = Collector(policy, test_envs, preprocess_fn=None)
# log
writer = SummaryWriter(os.path.join(logdir, 'MTL', 'ppo'))

def stop_fn(mean_rewards):
    # if env.env.spec.reward_threshold:
    #     return mean_rewards >= env.spec.reward_threshold
    # else:
    return False

# trainer
result = onpolicy_trainer(
    policy, train_collector, test_collector, epoch,
    step_per_epoch, collect_per_step, repeat_per_collect,
    test_num, batch_size, stop_fn=stop_fn, writer=writer)
if __name__ == '__main__':
    pprint.pprint(result)
    # Let's watch its performance!
    env = create_env(hidden_dim, data_path, train_size, size_task_class, lr, once_iter, max_iter)
    collector = Collector(policy, env, preprocess_fn=None)
    result = collector.collect(n_step=2000, render=None)
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')
