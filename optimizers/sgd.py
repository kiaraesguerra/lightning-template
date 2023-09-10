import torch
        
def sgd(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.max_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    return optimizer