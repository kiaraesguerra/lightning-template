from torchmetrics import BalancedAccuracy


def balanced_accuracy_metric(args):
    return BalancedAccuracy(num_classes=args.num_classes, average=args.average)
