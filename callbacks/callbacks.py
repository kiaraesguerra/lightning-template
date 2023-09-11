import callbacks


def get_callback(args):
    
    callbacks_list = []
    
    callbacks_name_list = [name for flag, name in {
        args.checkpoint: 'checkpoint',
        args.early_stopping: 'early_stopping',
        args.summary: 'summary',
        args.prune: 'prune',
        args.low_rank: 'low_rank',
        args.ptq: 'ptq',
        args.qat: 'qat',
    }.items() if flag]
    
    
    for name in callbacks_name_list:
        callbacks_list.append(callbacks.__dict__[name](args))


    return callbacks_list
