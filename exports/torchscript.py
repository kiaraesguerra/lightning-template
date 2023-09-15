import torch


def torchscript_export(model, args):
    scripted_model = torch.jit.script(model)
    scripted_model_path = (
        f"{args.dirpath}/{args.experiment_name}/{args.experiment_name}_scripted.pt"
    )
    scripted_model.save(scripted_model_path)
