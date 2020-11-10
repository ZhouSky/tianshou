import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from myenv import MyEnv
from datautil import read_data_from_file, MTDataset_Split, MTDataset


def generate_task_index(task_interval):
    task_ind = np.zeros((task_interval[-1]), dtype=np.int32)
    for i in range(task_interval.size - 1):
        task_ind[task_interval[i]: task_interval[i + 1]] = i
    return task_ind


def compute_errors(output, task_ind, label, num_task):
    num_ins = np.zeros([num_task])
    errors = np.zeros([num_task + 1])
    for i in range(output.shape[0]):
        num_ins[task_ind[i]] += 1
        if np.argmax(output[i, :]) != label[i]:
            errors[task_ind[i]] += 1
    for i in range(num_task):
        errors[i] = errors[i] / num_ins[i]
    errors[-1] = np.mean(errors[0:num_task])
    return errors


def test_net(net, testdata, testlabel, test_task_interval):
    task_ind = generate_task_index(test_task_interval)
    outputs = []
    with torch.no_grad():
        for a in range(testdata.shape[0]):
            outputs.append(
                net(
                    torch.tensor(testdata[a], dtype=torch.float),
                    task_ind[a],
                )
            )
        output = torch.stack(outputs, 0)
        test_error = compute_errors(
            output.numpy(), task_ind, testlabel, test_task_interval.size - 1
        )

        return test_error


class MTN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_task):
        super(MTN, self).__init__()
        shared_models = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
        for i in range(1, len(hidden_dim)):
            shared_models.extend(
                [nn.Linear(hidden_dim[i - 1], hidden_dim[i]), nn.ReLU()]
            )
        self.shared_models = nn.ModuleList(shared_models)
        self.task_models = nn.ModuleList(
            [nn.Linear(hidden_dim[-1], output_dim) for _ in range(num_task)]
        )
        self.num_task = num_task
        self.num_class = output_dim
        self.reset()

    def forward(self, x, task_index):
        out = x
        for _, m in enumerate(self.shared_models):
            out = m(out)
        return F.softmax(self.task_models[task_index](out), dim=-1)

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)


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


if __name__ == '__main__':
    lr = 1e-3
    size_task_class = 4  # means batch size: s_t_c * n_t * n_c
    hidden_dim = [600]
    train_size = 0.8
    once_iter = 50
    max_iter = 500

    data, label, task_interval, num_task, num_class = read_data_from_file(
        "./Office_Caltech_alexnet.txt"
    )
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
