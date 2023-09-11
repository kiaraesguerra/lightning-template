import exports


def export_model(model, args):
    exports_list = [
        name
        for flag, name in {args.onnx: "onnx_export", args.torch: "torch_export"}.items()
        if flag
    ]
    for export_name in exports_list:
        model = exports.__dict__[export_name](model, args)
