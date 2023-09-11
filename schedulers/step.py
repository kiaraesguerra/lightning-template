from torch.optim import lr_scheduler


def step_scheduler(optimizer, args):
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
        last_epoch=args.last_epoch,
    )
    return scheduler
