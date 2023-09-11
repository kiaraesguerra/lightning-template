from pytorch_lightning.callbacks import ModelCheckpoint


def checkpoint(args):
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        save_top_k=args.save_top_k,
        auto_insert_metric_name=False,
        save_last=args.save_last,
        filename=args.filename,
        save_on_train_epoch_end=True,
        dirpath=f"{args.dirpath}/{args.experiment_name}",
        verbose=True,
        mode="max",
    )
    return checkpoint_callback
    
        