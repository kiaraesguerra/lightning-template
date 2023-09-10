import torch 

def radam(model, args):
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    return optimizer