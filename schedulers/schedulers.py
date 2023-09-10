import warmup_scheduler
from schedulers import *
import schedulers


def get_scheduler(optimizer, args):
    try:
        scheduler_cls = schedulers.__dict__[args.scheduler](optimizer, args)
        
    except KeyError:
        raise ValueError(f"Invalid scheduler name: {args.optimizer}")    
    
    if args.warmup_epochs:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.warmup_epochs,
            after_scheduler=scheduler_cls,
        )
    else:
        scheduler = scheduler_cls

    return scheduler






