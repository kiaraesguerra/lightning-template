import torchvision


def cifar10(args, train_transform, test_transform):
    train_ds = torchvision.datasets.CIFAR10(
        "./datasets", train=True, transform=train_transform, download=True
    )
    test_ds = torchvision.datasets.CIFAR10(
        "./datasets", train=False, transform=test_transform, download=True
    )
    validate_ds = None
    args.num_classes = 10
    args.in_channels = 3

    return train_ds, validate_ds, test_ds
