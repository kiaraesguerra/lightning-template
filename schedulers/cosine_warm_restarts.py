from torch.optim import lr_scheduler


def cosine_warm_restarts_scheduler(optimizer, args):
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs, T_mult=args.min_lr
    )
    return scheduler
