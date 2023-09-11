import torch


def torchscript_export(model, args):
    scripted_model = torch.jit.script(model)
    # Specify the file path where you want to save the model
    scripted_model_path = "your_scripted_model.pt"

    # Save the TorchScript module to the specified file
    scripted_model.save(scripted_model_path)
