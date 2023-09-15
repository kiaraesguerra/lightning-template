from torchmetrics import Precision


def precision_metric(args):
    return Precision(num_classes=args.num_classes, average=args.average)
