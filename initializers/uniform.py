import torch.nn as nn

def UniformInit(model, a=0.0, b=1.0):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, a=a, b=b)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model