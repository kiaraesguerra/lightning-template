import torch
import torch.nn as nn
from itertools import product
from initializers.base import Base

class Delta_Module(Base):
    def __init__(
        self,
        module: nn.Module,
        gain: int = 1,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str = "relu",
        same_mask: bool = False,
        in_channels: int = 3,
        num_classes: int = 100,
    ):
        self.module = module
        self.sparsity = sparsity
        self.degree = degree
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.method = method
        self.activation = activation
        self.same_mask = same_mask
        self.gain = gain

    def _delta(self) -> torch.tensor:
        weights = self._ortho_generator()
        delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        delta_weights[:, :, 1, 1] = weights

        return delta_weights

    def __call__(self):
        return self._delta()


def Delta_Constructor(module, **kwargs):
    return Delta_Module(module, **kwargs)()


def DeltaInit(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            vals = Delta_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(module, "weight", torch.abs(vals[1]))
                
            else:
                module.weight = nn.Parameter(vals)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model
