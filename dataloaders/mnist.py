import torchvision


def mnist(args, train_transform, test_transform):
    train_ds = torchvision.datasets.MNIST(
        "./datasets", train=True, transform=train_transform, download=True
    )
    test_ds = torchvision.datasets.MNIST(
        "./datasets", train=False, transform=test_transform, download=True
    )
    validate_ds = None
    args.num_classes = 10
    args.in_channels = 1
    
    return train_ds, validate_ds, test_ds


