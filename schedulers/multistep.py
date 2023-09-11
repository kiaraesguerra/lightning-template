from torch.optim import lr_scheduler


def multistep_scheduler(optimizer, args):
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    return scheduler
