from abc import ABC

import torch.nn as nn
import numpy as np


class BaseNet(nn.Module, ABC):
    @staticmethod
    def seed(seed=0):
        np.random.seed(seed)
        return [seed]

    def reset(self, std=1e-3):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)


class STNS(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, num_task=1):
        super(STNS, self).__init__()

        models = []
        for t in range(num_task):
            sub_models = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
            for i in range(1, len(hidden_dim)):
                sub_models.extend(
                    [nn.Linear(hidden_dim[i - 1], hidden_dim[i]), nn.ReLU()]
                )
            sub_models.extend(
                [nn.Linear(hidden_dim[-1], output_dim), nn.Softmax(dim=-1)]
            )
            models.append(nn.Sequential(*sub_models))
        self.models = nn.ModuleList(models)

    def forward(self, x, task_index=0):
        return self.models[task_index](x)


class MTN(BaseNet):
    def __init__(self, input_dim: int, hidden_dim: list, output_dim: int, num_task: int):
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
        out = self.shared_models(x)
        return self.task_models[task_index](out)
