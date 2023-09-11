import torch


def rmsprop(model, args):
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        alpha=args.alpha,
        eps=args.eps,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        centered=args.centered,
    )
    return optimizer
