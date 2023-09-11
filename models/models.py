from models.mlp import MLP
import timm


def get_model(args):
    if args.model == "mlp":
        model = MLP(
            image_size=args.image_size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            activation=args.activation,
            num_layers=args.num_layers,
            hidden_width=args.width,
        )
    else:
        model = timm.create_model(
            args.model, pretrained=args.pretrained, num_classes=args.num_classes
        )

    return model
