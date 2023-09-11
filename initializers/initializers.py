import torch.nn as nn
import initializers


def get_weight_init(model, args):
    try:
        model = initializers.__dict__[args.weight_init](model, args)

    except KeyError:
        raise ValueError(f"Invalid initializer name: {args.weight_init}")

    return model
