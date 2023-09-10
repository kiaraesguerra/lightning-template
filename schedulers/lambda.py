import torch.optim.lr_scheduler as lr_scheduler


def lambda(optimizer, args):
    scheduler = lr_scheduler.LambdaLR(optimizer)
    return scheduler
