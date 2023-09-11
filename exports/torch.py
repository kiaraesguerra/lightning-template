import torch


def torch_export(model, args):
    torch.save(
        model.model.state_dict(),
        f"{args.dirpath}/{args.experiment_name}/{args.experiment_name}.pt",
    )
