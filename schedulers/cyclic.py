import torch.optim.lr_scheduler as lr_scheduler


def cyclic(optimizer, args):
    scheduler = lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        step_size_up=args.step_size_up,
        step_size_down=args.step_size_down,
        mode=args.mode,
        gamma=args.gamma,
        scale_fn=args.scale_fn,
        scale_mode=args.scale_mode,
        cycle_momentum=args.cycle_momentum,
        base_momentum=args.base_momentum,
        max_momentum=args.max_momentum,
        last_epoch=args.last_epoch,
    )
    return scheduler