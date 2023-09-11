import torch


def onnx_export(model, args):
    torch.onnx.export(
        model,
        args.sample_input,
         f"{args.dirpath}/{args.experiment_name}/model.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
