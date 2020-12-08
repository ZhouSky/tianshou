class EnvArgs:
    def __init__(self, reward_fn, state_fn):
        self.lr = 1e-3
        self.size_task_class = 15  # means batch size: s_t_c * n_t * n_c
        self.hidden_dim = [600]
        self.train_size = 0.8
        self.iter_step = 10
        self.max_iter = 0  # only one make sense
        self.max_epoch = 500  # only one make sense
        self.data_path = "./Office_Caltech_alexnet.txt"
        self.reward_fn = reward_fn
        self.state_fn = state_fn


class PPOArgs:
    gamma = 0.99
    max_grad_norm = 0.5
    eps_clip = 0.2
    vf_coef = 0.5
    ent_coef = 0.01


class TrainArgs:
    lr = 1e-3
    layer_num = 1
    seed = 0
    training_num = 6
    test_num = 1
    buffer_size = 4096
    logdir = 'log'
    epoch =27  # each epoch will test policy once
    step_per_epoch = 300  # num of policy-training per epoch for policy train
    collect_per_step = 6  # num of eps per iter for policy train
    repeat_per_collect = 10  # repeat train
    batch_size = 500  # num of step-data per policy-training
