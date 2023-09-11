import torch.nn as nn


def cross_entropy_loss(args):
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
