from torch.optim import lr_scheduler


def exponential_scheduler(optimizer, args):
    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.gamma,
        last_epoch=args.last_epoch,
    )
    return scheduler
