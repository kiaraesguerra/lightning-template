import torch
import optimizers


def get_optimizer(model, args):
    try:
        optimizer = optimizers.__dict__[args.optimizer](model, args)
        
    except KeyError:
        raise ValueError(f"Invalid optimizer name: {args.optimizer}")    
    
    return optimizer




