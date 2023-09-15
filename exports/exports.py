import exports


def export_model(model, args):
    defaults = ["torch"]
    exports_name_list = (
        defaults if args.exports is None else set(args.exports + defaults)
    )

    for export in exports_name_list:
        exports.__dict__[export + "_export"](model, args)
