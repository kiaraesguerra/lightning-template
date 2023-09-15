from torchmetrics import Recall


def recall_metric(args):
    return Recall(num_classes=args.num_classes, average=args.average)
