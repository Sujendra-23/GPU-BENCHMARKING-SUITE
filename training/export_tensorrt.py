"""Exports a trained PyTorch model to ONNX, then builds a TensorRT engine
at a target precision (fp32/fp16/int8), for use by harness/benchmark.py's
build_tensorrt_runner().

Requires torch, onnx, and the TensorRT Python bindings on a machine with an
NVIDIA GPU - not available in the sandbox this repo was built in.

    python export_tensorrt.py --checkpoint model.pt --precision fp16 \
        --batch-size 32 --out engines/resnet50_fp16_bs32.trt
"""
from __future__ import annotations

import argparse
import os


def export_to_onnx(checkpoint_path: str, onnx_path: str, batch_size: int) -> None:
    import torch

    model = torch.load(checkpoint_path, map_location="cpu")
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=17,
    )
    print(f"exported ONNX model to {onnx_path}")


def build_tensorrt_engine(onnx_path: str, engine_path: str, precision: str, batch_size: int) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB scratch space for kernel selection

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # A real INT8 build needs a calibrator with representative data;
        # omitted here since calibration data is deployment-specific.
        raise NotImplementedError("INT8 requires an IInt8Calibrator wired to real calibration data")

    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"built TensorRT engine ({precision}, batch={batch_size}) -> {engine_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx_path = args.out.replace(".trt", ".onnx")

    export_to_onnx(args.checkpoint, onnx_path, args.batch_size)
    build_tensorrt_engine(onnx_path, args.out, args.precision, args.batch_size)


if __name__ == "__main__":
    main()
