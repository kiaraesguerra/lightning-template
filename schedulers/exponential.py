import torch.optim.lr_scheduler as lr_scheduler

def exponential(optimizer, args):
    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.gamma,
        last_epoch=args.last_epoch,
    )
    return scheduler