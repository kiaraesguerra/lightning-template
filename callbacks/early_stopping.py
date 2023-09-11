from pytorch_lightning.callbacks import EarlyStopping


def early_stopping(args):
    return EarlyStopping(
        monitor=args.monitor,
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=args.verbose,
        mode=args.mode,
        strict=args.strict,
    )