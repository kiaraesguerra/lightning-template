import exports


def export_model(model, args):
    defaults = ["torch"]
    exports_name_list = set(args.exports + defaults)

    for export in exports_name_list:
        exports.__dict__[export + "_export"](model, args)
