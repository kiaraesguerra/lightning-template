import argparse
import torch
from pytorch_lightning import Trainer
from dataloaders.dataloaders import get_dataloader
from models.models import get_model
from initializers.initializers import get_weight_init
from utils.train_utils import get_plmodule
from callbacks.callbacks import get_callback
from loggers.loggers import get_logger
from exports.exports import export_model
from lrs.lrs_init_call import get_ls_init
from utils.pruning_utils import measure_sparsity, remove_parameters


parser = argparse.ArgumentParser(description="PyTorch Lightning Trainer")

# Model Initialization
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--norm-off", action="store_true")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--weight-init", type=str, default="OrthoInit")
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument(
    "--pretrained-path",
    type=str,
    default=None,
    metavar="Path to the user-trained model",
)

# For custom architectures, i.e. MLP
parser.add_argument("--num-layers", type=int, default=5)
parser.add_argument("--width", "--hidden-width", type=int, default=16)
parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])

# Dataloader and augmentation
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--eval-batch-size", type=int, default=100, metavar="N")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=3407)

# Augmentations
parser.add_argument("--cutmix-prob", type=float, default=0.0)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--horizontal-flip", type=float, default=0)
parser.add_argument("--vertical-flip", type=float, default=0)
parser.add_argument("--random-rotations", type=float, default=0)
parser.add_argument("--color-jitter", type=float, default=0)

# Training
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--warmup-epochs", type=int, default=0)
parser.add_argument("--max-lr", type=float, default=1e-1)
parser.add_argument("--min-lr", type=float, default=1e-3)

# Scheduler settings
parser.add_argument("--scheduler", type=str, default="multistep")
parser.add_argument("--milestones", type=list, default=[100, 150])

# Criterion
parser.add_argument("--criterion", type=str, default="cross_entropy")
parser.add_argument("--label-smoothing", type=float, default=0.1)

# Optimizer settings
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=1.0e-08)
parser.add_argument("--nesterov", action="store_true")
parser.add_argument("--amsgrad", action="store_true")

# Adam hyperparameters
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)

# Regularization
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--drop-out", type=float, default=1e-4)

# Callbacks
parser.add_argument("--callbacks", nargs="+")
# Can add: "prune", "cutmix", "qat", "ptq", "low-rank", "summary", "early_stopping"

parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--save-last", action="store_true")
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--dirpath", type=str, default="results")
parser.add_argument("--filename", type=str, default="best")

# Pruning
parser.add_argument("--method", type=str, default="l1")
parser.add_argument("--speed-up", type=float, default=2.0)
parser.add_argument("--ch-sparsity", type=float, default=1.0)
parser.add_argument("--max-sparsity", type=float, default=1.0)

# Exports
parser.add_argument("--exports", nargs="+", help="<Required> Set flag")
# Can add: "onnx", "torchscript"


args = parser.parse_args()


if __name__ == "__main__":
    train_dl, validate_dl, test_dl = get_dataloader(args)
    model = get_model(args)

    # args.baseline_path should point to the .pt file
    if args.pretrained_path:
        model_checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(model_checkpoint["state_dict"])
    elif args.pretrained:
        pass
    else:
        model = get_weight_init(model, args)

    if args.lrs:
        model = get_ls_init(model, args)

    model = get_plmodule(model, args)
    callbacks = get_callback(args)
    logger = get_logger(args)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_dl, validate_dl)
    trainer.test(dataloaders=test_dl)

    model_sparsity, model_nonzeros = measure_sparsity(model)
    print(f"Model sparsity = {model_sparsity}, number of nonzeros = {model_nonzeros}")
    if callbacks:
        ckpt_path = [a for a in callbacks if "checkpoint" in str(a)][0].best_model_path
    model_checkpoint = torch.load(ckpt_path)

    model.load_state_dict(model_checkpoint["state_dict"])
    remove_parameters(model)
    args.sample_input = train_dl.dataset[0][0].unsqueeze(0)
    export_model(model, args)
