import torch.nn as nn
import criterions


def get_criterion(args):
    try:
        criterion = criterions.__dict__[args.criterion + '_loss'](args)
    except KeyError:
        raise ValueError(f"Invalid criterion name: {args.criterion}")

    return criterion
