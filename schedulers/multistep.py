import torch.optim.lr_scheduler as lr_scheduler


def multistep(optimizer, args):
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    return scheduler