from torch.optim import lr_scheduler


def cosine_scheduler(optimizer, args):
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    return scheduler
