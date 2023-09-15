import callbacks


def get_callback(args):
    defaults = ["checkpoint"]

    callbacks_name_list = (
        defaults if args.callbacks is None else set(args.callbacks + defaults)
    )

    callbacks_list = [
        callbacks.__dict__[callback + "_callback"](args)
        for callback in callbacks_name_list
    ]

    return callbacks_list
