from functools import partial
from pytorch_lightning.callbacks import Callback
import torch_pruning as tp
import torch
import torch.nn as nn


class Pruner(Callback):
    def __init__(
        self,
        method: str = "l1",
        speed_up: float = 2.0,
        ch_sparsity: float = 1.0,
        max_sparsity: float = 1.0,
    ):
        self.method = method
        self.speed_up = speed_up
        self.ch_sparsity = ch_sparsity
        self.max_sparsity = max_sparsity
        self.current_speed_up = 1.0
        self.num_classes = 10

        self.example_inputs = torch.randn(3, 3, 32, 32).to("cuda")
        self.reg = 5e-4
        self.global_pruning = True

    def _get_pruner(self, model):
        self.sparsity_learning = False
        if self.method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(
                tp.pruner.MagnitudePruner, global_pruning=self.global_pruning
            )
        elif self.method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(
                tp.pruner.MagnitudePruner, global_pruning=self.global_pruning
            )
        elif self.method == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(
                tp.pruner.MagnitudePruner, global_pruning=self.global_pruning
            )
        elif self.method == "slim":
            self.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(
                tp.pruner.BNScalePruner,
                reg=self.reg,
                global_pruning=self.global_pruning,
            )
        elif self.method == "group_norm":
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(
                tp.pruner.GroupNormPruner, global_pruning=self.global_pruning
            )
        elif self.method == "group_sl":
            self.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(
                tp.pruner.GroupNormPruner,
                reg=self.reg,
                global_pruning=self.global_pruning,
            )
        else:
            raise NotImplementedError

        unwrapped_parameters = []
        ignored_layers = []
        ch_sparsity_dict = {}

        for module in model.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and module.out_features == self.num_classes
            ):
                ignored_layers.append(module)
            elif (
                isinstance(module, torch.nn.modules.conv._ConvNd)
                and module.out_channels == self.num_classes
            ):
                ignored_layers.append(module)

        pruner = pruner_entry(
            model,
            self.example_inputs,
            importance=imp,
            iterative_steps=200,
            ch_sparsity=0.9,
            ch_sparsity_dict=ch_sparsity_dict,
            max_ch_sparsity=0.9,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
        return pruner

    def progressive_pruning(self, model):
        model.eval()
        base_ops, _ = tp.utils.count_ops_and_params(
            model.model, example_inputs=self.example_inputs
        )

        while self.current_speed_up < self.speed_up:
            self.pruner.step()
            pruned_ops, _ = tp.utils.count_ops_and_params(
                model.model, example_inputs=self.example_inputs
            )
            self.current_speed_up = float(base_ops) / pruned_ops
        print(f"Current speed up: {self.current_speed_up}")

    def on_fit_start(self, trainer, pl_module):
        self.pruner = self._get_pruner(pl_module)
        self.progressive_pruning(pl_module)


def prune_callback(args):
    pruner = Pruner(
        method=args.method,
        speed_up=args.speed_up,
        ch_sparsity=args.ch_sparsity,
        max_sparsity=args.max_sparsity,
    )

    return pruner
