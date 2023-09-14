import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        activation: str = "relu",
        num_layers: int = 5,
        hidden_width: int = 128,
        batch_norm: bool = True,
    ):
        super(MLP, self).__init__()

        self.image_size = image_size
        self.num_input_channels = in_channels
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.input_layer = nn.Linear(
            image_size * image_size * in_channels, hidden_width
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_width, hidden_width, bias=False)
                for _ in range(num_layers - 1)
            ]
        )

        if self.batch_norm is True:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(hidden_width) for _ in range(num_layers - 1)]
            )
        self.output_layer = nn.Linear(hidden_width, num_classes)

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size * self.num_input_channels)
        x = self.input_layer(x)
        x = self.activation(x)
        for layer, batchnorm in zip(self.hidden_layers, self.bn_layers):
            x = layer(x)
            if self.batch_norm is True:
                x = batchnorm(x)
            x = self.activation(x)
        x = self.output_layer(x)

        return x


def mlp(args):
    return MLP(
        image_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        activation=args.activation,
        num_layers=args.num_layers,
        hidden_width=args.width,
    )
