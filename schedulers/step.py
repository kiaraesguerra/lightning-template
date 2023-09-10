import torch.optim.lr_scheduler as lr_scheduler

def step(optimizer, args):
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
        last_epoch=args.last_epoch,
    )
    return scheduler