import numpy as np
import torch


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
