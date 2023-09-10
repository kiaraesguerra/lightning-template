import torch.optim.lr_scheduler as lr_scheduler

def cosine(optimizer, args):
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    return scheduler