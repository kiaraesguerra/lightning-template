from torch.optim import lr_scheduler


def reduce_on_plateau_scheduler(optimizer, args):
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=args.mode,
        factor=args.factor,
        patience=args.patience,
        verbose=args.verbose,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        cooldown=args.cooldown,
        min_lr=args.min_lr,
        eps=args.eps,
    )
    return scheduler
