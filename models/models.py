from models.mlp import MLP
import timm
import models


def get_model(args):
    if "mlp" in args.model:
        model = models.__dict__[args.model](args)
    else:
        model = timm.create_model(
            args.model, pretrained=args.pretrained, num_classes=args.num_classes
        )

    return model
