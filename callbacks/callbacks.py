import callbacks


def get_callback(args):
    callbacks_list = []

    callbacks_name_list = [
        name
        for flag, name in {
            args.checkpoint: "checkpoint_callback",
            args.early_stopping: "early_stopping_callback",
            args.summary: "summary_callback",
            args.prune: "prune_callback",
            args.low_rank: "low_rank_callback",
            args.ptq: "ptq_callback",
            args.qat: "qat_callback",
            args.cutmix: "cutmix_callback",
        }.items()
        if flag
    ]

    for name in callbacks_name_list:
        callbacks_list.append(callbacks.__dict__[name](args))

    return callbacks_list
