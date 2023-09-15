from torchmetrics import Accuracy


def accuracy_metric(args):
    return Accuracy(num_classes=args.num_classes, average=args.average)
