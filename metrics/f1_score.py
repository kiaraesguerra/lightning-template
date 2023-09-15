from torchmetrics import F1_Score


def f1_score_metric(args):
    return F1_Score(num_classes=args.num_classes, average=args.average)
