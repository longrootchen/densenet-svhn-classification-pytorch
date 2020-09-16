import os
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


class Config:

    workers = 4
    gpu = 'cuda:0'

    train_root = os.path.join(os.pardir, 'SVHN', 'TrainValid', 'train')
    valid_root = os.path.join(os.pardir, 'SVHN', 'TrainValid', 'valid')

    epochs = 40
    save_dir = os.path.join(os.curdir, 'checkpoints')
    log_dir = os.path.join(os.curdir, 'logs')
    verbose = True

    num_classes = 10
    arch = 'densenet250_k24_bc_svhn'
    resume = os.path.join(os.curdir, 'checkpoints', 'last_checkpoint.pth')
    batch_size = 64

    CriterionClass = CrossEntropyLoss
    OptimizerClass = SGD
    no_decays = ['bn', 'bias']
    weight_decay = 0.0001
    optimizer_params = dict(lr=0.1, momentum=0.9)
    SchedulerClass = MultiStepLR
    scheduler_params = dict(gamma=0.1, milestones=[20, 30])
