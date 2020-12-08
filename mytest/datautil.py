import re
import numpy as np


# 分批，numpy
class MTDataset:
    def __init__(self, data, label, task_interval, num_class, batch_size):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.ravel(label)
        self.task_interval = np.ravel(task_interval)
        self.num_task = task_interval.size - 1
        self.num_class = num_class
        self.batch_size = batch_size
        self.__build_index__()

    def copy(self):
        return MTDataset(self.data, self.label, self.task_interval, self.num_class, self.batch_size)

    def __build_index__(self):
        index_list = []
        for i in range(self.num_task):
            start = self.task_interval[i]
            end = self.task_interval[i + 1]
            for j in range(self.num_class):
                index_list.append(
                    np.arange(start, end)[np.where(self.label[start:end] == j)[0]]
                )
        # [num_task, num_class, num_index], index in data/label
        self.index_list = index_list
        # count number of selected samples
        self.counter = np.zeros([self.num_task * self.num_class], dtype=np.int32)

    def get_next_batch(self):
        sampled_data = np.zeros(
            [self.batch_size * self.num_class * self.num_task, self.data_dim],
            dtype=np.float32,
        )
        # 这个似乎和最后一个作用重合
        sampled_label = np.zeros(
            [self.batch_size * self.num_class * self.num_task, self.num_class],
            dtype=np.int32,
        )
        sampled_task_ind = np.zeros(
            [self.batch_size * self.num_class * self.num_task], dtype=np.int32
        )
        sampled_label_ind = np.zeros(
            [self.batch_size * self.num_class * self.num_task], dtype=np.int32
        )
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                sampled_ind = range(
                    cur_ind * self.batch_size, (cur_ind + 1) * self.batch_size
                )
                sampled_task_ind[sampled_ind] = i
                sampled_label_ind[sampled_ind] = j
                sampled_label[sampled_ind, j] = 1
                # 总样本量不够，则随机重复选择补全
                if task_class_index.size < self.batch_size:
                    sampled_data[sampled_ind, :] = self.data[
                                                   np.concatenate(
                                                       (
                                                           task_class_index,
                                                           task_class_index[
                                                               np.random.randint(
                                                                   0,
                                                                   high=task_class_index.size,
                                                                   size=self.batch_size - task_class_index.size,
                                                               )
                                                           ],
                                                       )
                                                   ),
                                                   :,
                                                   ]
                    np.random.shuffle(self.index_list[cur_ind])
                # 剩余样本够，则向后选
                elif self.counter[cur_ind] + self.batch_size < task_class_index.size:
                    sampled_data[sampled_ind, :] = self.data[
                                                   task_class_index[
                                                   self.counter[cur_ind]: self.counter[cur_ind]
                                                                          + self.batch_size
                                                   ],
                                                   :,
                                                   ]
                    self.counter[cur_ind] = self.counter[cur_ind] + self.batch_size
                # 选完，则选后batch_size个，然后counter重记，打乱顺序
                else:
                    sampled_data[sampled_ind, :] = self.data[
                                                   task_class_index[-self.batch_size:], :
                                                   ]
                    self.counter[cur_ind] = 0
                    np.random.shuffle(self.index_list[cur_ind])
        # [batch_class_task, dim], [bct, class], [bct], [bct]
        return sampled_data, sampled_label, sampled_task_ind, sampled_label_ind


# 划分训练集和测试集，numpy
class MTDataset_Split:
    def __init__(self, data, label, task_interval, num_class):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.ravel(label)
        self.task_interval = np.ravel(task_interval)
        self.num_task = task_interval.size - 1
        self.num_class = num_class
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        self.num_class_ins = np.zeros([self.num_task, self.num_class])
        for i in range(self.num_task):
            start = self.task_interval[i]
            end = self.task_interval[i + 1]
            for j in range(self.num_class):
                # np.where在没有x和y时，返回tuple，里面是包含索引的list,这里取list,这里由于对label切片了，得到的索引不是原label的
                index_array = np.where(self.label[start:end] == j)[0]
                # 获得每个任务每类的样本数
                self.num_class_ins[i, j] = index_array.size
                # 任务数*种类数个list，各个任务各个类的样本在data中的索引按顺序存在index_list里
                index_list.append(np.arange(start, end)[index_array])
        self.index_list = index_list

    def split(self, train_size):
        if train_size < 1:
            train_num = np.ceil(self.num_class_ins * train_size).astype(np.int32)
        else:
            train_num = (
                    np.ones([self.num_task, self.num_class], dtype=np.int32) * train_size
            )
            # 对每个任务每个种类，至少留一个样本训练，然后至少留10个样本测试
            train_num = int(
                np.maximum(1, np.minimum(train_num, self.num_class_ins - 10))
            )
        traindata = np.zeros([0, self.data_dim], dtype=np.float32)
        testdata = np.zeros([0, self.data_dim], dtype=np.float32)
        trainlabel = np.zeros([0], dtype=np.int32)
        testlabel = np.zeros([0], dtype=np.int32)
        train_task_interval = np.zeros([self.num_task + 1], dtype=np.int32)
        test_task_interval = np.zeros([self.num_task + 1], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                # 取到i任务j类的样本在data中对应的索引的list
                task_class_index = self.index_list[cur_ind]
                # 打乱顺序
                np.random.shuffle(task_class_index)
                train_index = task_class_index[0: train_num[i, j]]
                test_index = task_class_index[train_num[i, j]:]
                # 从data中取出放入traindata
                traindata = np.concatenate(
                    (traindata, self.data[train_index, :]), axis=0
                )
                trainlabel = np.concatenate(
                    (trainlabel, np.ones([train_index.size], dtype=np.int32) * j),
                    axis=0,
                )
                testdata = np.concatenate((testdata, self.data[test_index, :]), axis=0)
                testlabel = np.concatenate(
                    (testlabel, np.ones([test_index.size], dtype=np.int32) * j),
                    axis=0,
                )
            # 对应的是traindata和testdata和trainlabel和testlabel
            train_task_interval[i + 1] = trainlabel.size
            test_task_interval[i + 1] = testlabel.size
        # arrays
        return (
            traindata,
            trainlabel,
            train_task_interval,
            testdata,
            testlabel,
            test_task_interval,
        )


def read_data_from_file(filename):
    file = open(filename, "r")
    contents = file.readlines()
    file.close()
    num_task = int(contents[0])
    num_class = int(contents[1])
    task_interval = np.array(list(map(int, re.split(",", contents[2]))))
    data = np.array([list(map(float, re.split(',', contents[pos]))) for pos in range(3, len(contents) - 1)])
    label = np.array(list(map(int, re.split(',', contents[-1])))).ravel()
    return data, label, task_interval, num_task, num_class


def testReaderSpliterBatcher():
    batch_size = 10
    data, label, task_interval, num_task, num_class = read_data_from_file(
        "./Office_Caltech_alexnet.txt"
    )
    print(data.shape, label.shape, task_interval, num_task, num_class)
    print("--------------")
    data_split = MTDataset_Split(data, label, task_interval, num_class)
    (
        traindata,
        trainlabel,
        train_task_interval,
        testdata,
        testlabel,
        test_task_interval,
    ) = data_split.split(0.8)
    print(
        traindata.shape,
        trainlabel.shape,
        train_task_interval,
        testdata.shape,
        testlabel.shape,
        test_task_interval,
    )
    trainiter = MTDataset(
        traindata, trainlabel, train_task_interval, num_class, batch_size
    )
    task_index, class_index = (
        np.random.randint(0, num_task, 1)[0],
        np.random.randint(0, num_class, 1)[0],
    )
    data, label, sampled_task_ind, sampled_label_ind = trainiter.get_next_batch()
    assert (
            (
                    sampled_task_ind[
                    task_index
                    * num_class
                    * batch_size: (task_index + 1)
                                  * num_class
                                  * batch_size
                    ]
                    == task_index
            ).all()
            and (
                    sampled_label_ind[
                    (task_index * num_class + class_index)
                    * batch_size: (task_index * num_class + class_index + 1)
                                  * batch_size
                    ]
                    == class_index
            ).all()
            and data.shape[0] == batch_size * num_task * num_class
            and label.shape[0] == data.shape[0]
    )


if __name__ == '__main__':
    testReaderSpliterBatcher()
