"""Build/evaluate calibrated ResNet-18/50 FP32, FP16 and INT8 TensorRT engines.

Run from the repository root: python -m training.export_tensorrt --suite
--calibration-dir /data/imagenet/train --eval-dir /data/imagenet/val.
INT8 uses representative images to produce explicit ONNX Q/DQ, not random ranges.
The legacy --checkpoint/--precision/--out single-engine interface is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.measurement import metadata, positive_int, synchronized_time, write_json
from harness.tensorrt_utils import TensorRTRunner, build_engine, require_cuda


def load_model(name, checkpoint=None):
    import torch
    import torchvision.models as models
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(saved, torch.nn.Module):
            return saved.eval()
        model = getattr(models, name)(weights=None)
        state = saved.get("state_dict", saved)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        return model.eval()
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[name]
    return getattr(models, name)(weights=weights).eval()


def export_model(model, inputs, onnx_path, input_names=("input",), output_names=("output",)):
    import torch
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    import inspect
    options = {"dynamo": False}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        options["external_data"] = False
    with torch.inference_mode():
        torch.onnx.export(model, inputs, str(onnx_path), input_names=list(input_names),
                          output_names=list(output_names), opset_version=17, **options)


def export_to_onnx(checkpoint_path, onnx_path, batch_size, model_name="resnet50"):
    import torch
    export_model(load_model(model_name, checkpoint_path), torch.randn(batch_size, 3, 224, 224), onnx_path)


def image_dataset(root, model_name, class_index=None, labels_required=True):
    from torchvision import datasets, models
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[model_name]
    dataset = datasets.ImageFolder(root, transform=weights.transforms())
    if labels_required:
        if class_index:
            mapping = json.loads(Path(class_index).read_text())
            if any(c not in mapping for c in dataset.classes):
                raise ValueError("class-index must map every directory name to an ImageNet logit index")
            if any(type(mapping[c]) is not int or not 0 <= mapping[c] < 1000 for c in dataset.classes):
                raise ValueError("ImageNet indices must be integers in [0,999]")
            remap = {index: mapping[name] for name, index in dataset.class_to_idx.items()}
            dataset.samples = [(path, remap[label]) for path, label in dataset.samples]
            dataset.targets = [label for _, label in dataset.samples]
        elif len(dataset.classes) != 1000 or not all(c.startswith("n") and c[1:].isdigit() for c in dataset.classes):
            raise ValueError("Use all 1000 ImageNet wnid directories or supply --class-index for a labeled subset")
    return dataset


def subset(dataset, count, seed):
    import torch
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    return torch.utils.data.Subset(dataset, indices)


def dataset_manifest(dataset):
    records = [{"path": str(Path(dataset.dataset.samples[i][0]).resolve()),
                "label": dataset.dataset.samples[i][1],
                "sha256": hashlib.sha256(Path(dataset.dataset.samples[i][0]).read_bytes()).hexdigest()}
               for i in dataset.indices]
    return records


def calibration_batches(dataset, batch_size):
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, _ in loader:
        if len(images) < batch_size:
            images = torch.cat([images, images[-1:].repeat(batch_size - len(images), 1, 1, 1)])
        yield {"input": images.numpy()}


def quantize_onnx(source, target, dataset, batch_size):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter(calibration_batches(dataset, batch_size))
        def get_next(self):
            return next(self.batches, None)

    prepared = Path(target).with_suffix(".prepared.onnx")
    quant_pre_process(str(source), str(prepared), skip_optimization=False)
    import onnx
    first_conv = next(n.name for n in onnx.load(str(prepared)).graph.node if n.op_type == "Conv")
    quantize_static(str(prepared), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.Entropy,
                    nodes_to_exclude=[first_conv], calibration_providers=["CUDAExecutionProvider"],
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False, "DedicatedQDQPair": True})


def build_tensorrt_engine(onnx_path, engine_path, precision, batch_size):
    return build_engine(onnx_path, engine_path, precision)


def accuracy(fn, dataset, batch_size):
    import torch
    correct1 = correct5 = total = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, labels in loader:
            valid = len(images)
            if valid < batch_size:
                images = torch.cat([images, images[-1:].repeat(batch_size - valid, 1, 1, 1)])
            logits = fn(images.cuda())[:valid].float()
            labels = labels.cuda()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite model output during accuracy evaluation")
            top = logits.topk(5, dim=-1).indices
            correct1 += (top[:, 0] == labels).sum().item()
            correct5 += (top == labels[:, None]).any(dim=1).sum().item()
            total += valid
    return {"samples": total, "top1_percent": 100 * correct1 / total, "top5_percent": 100 * correct5 / total}


def suite(args):
    torch = require_cuda()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.out_dir)
    result = {"metadata": metadata(["torch", "torchvision", "tensorrt", "onnxruntime", "onnx"]),
              "config": vars(args), "dataset_label": args.dataset_label, "accuracy_method": "held-out labeled images; original ImageNet-1000 logits; identical images/preprocessing per precision",
              "int8_method": "GPU entropy-calibrated symmetric per-channel ONNX Q/DQ; stem convolution/biases unquantized; FP16/FP32 fallback",
              "models": {}}
    for name in args.models.split(','):
        if name not in ("resnet18", "resnet50"):
            raise ValueError("models must be resnet18 and/or resnet50")
        calibration = subset(image_dataset(args.calibration_dir, name, labels_required=False), args.calibration_samples, args.seed)
        evaluation = subset(image_dataset(args.eval_dir, name, args.class_index), args.eval_samples, args.seed + 1)
        cal_manifest, eval_manifest = dataset_manifest(calibration), dataset_manifest(evaluation)
        if {r['sha256'] for r in cal_manifest} & {r['sha256'] for r in eval_manifest}:
            raise ValueError("Calibration and evaluation images overlap; use held-out evaluation data")
        write_json(root / f"{name}-data.json", {"calibration": cal_manifest, "evaluation": eval_manifest})
        model = load_model(name).cuda()
        x = next(calibration_batches(evaluation, args.batch_size))["input"]
        x = torch.from_numpy(x).cuda()
        fp32_onnx = root / f"{name}_fp32_bs{args.batch_size}.onnx"
        export_model(model, x, fp32_onnx)
        torch_accuracy = accuracy(model, evaluation, args.batch_size)
        rows = []
        for precision in ("fp32", "fp16", "int8"):
            source = fp32_onnx
            if precision == "int8":
                source = root / f"{name}_int8_bs{args.batch_size}.onnx"
                quantize_onnx(fp32_onnx, source, calibration, args.batch_size)
            path = Path(args.engine_dir) / f"{name}_{precision}_bs{args.batch_size}.trt"
            start = time.perf_counter()
            build_engine(source, path, precision, args.workspace_gib)
            build_seconds = time.perf_counter() - start
            engine = TensorRTRunner(path)
            fn = lambda images: engine(input=images)["output"]
            acc = accuracy(fn, evaluation, args.batch_size)
            timing = synchronized_time(lambda: fn(x), args.warmup, args.iterations)
            inspection = root / f"{name}-{precision}-layers.json"
            write_json(inspection, engine.inspect())
            row = {"precision": precision, "engine": str(path), "engine_size_bytes": path.stat().st_size,
                   "build_seconds": build_seconds, "accuracy": acc, "latency": timing,
                   "throughput_images_per_second": args.batch_size * 1000 / timing['mean_ms'],
                   "layer_inspection": str(inspection)}
            rows.append(row)
            base = rows[0]
            row.update({"top1_delta_pp_vs_fp32": acc['top1_percent'] - base['accuracy']['top1_percent'],
                        "latency_speedup_vs_fp32": base['latency']['median_ms'] / timing['median_ms'],
                        "engine_size_ratio_vs_fp32": row['engine_size_bytes'] / base['engine_size_bytes']})
            print(f"{name} {precision}: top1={acc['top1_percent']:.2f}%, median={timing['median_ms']:.3f} ms, {path.stat().st_size / (1 << 20):.2f} MiB")
            result['models'][name] = {"torch_fp32_accuracy": torch_accuracy, "results": rows}
            write_json(root / "benchmark.json", result)
            del engine
        del model
        torch.cuda.empty_cache()
    lines = ["# ResNet TensorRT quantization", "", "| model | precision | top-1 % | delta pp vs FP32 | median ms | speedup | engine MiB | size / FP32 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, report in result['models'].items():
        for r in report['results']:
            lines.append(f"| {name} | {r['precision']} | {r['accuracy']['top1_percent']:.2f} | {r['top1_delta_pp_vs_fp32']:.2f} | {r['latency']['median_ms']:.3f} | {r['latency_speedup_vs_fp32']:.2f} | {r['engine_size_bytes'] / (1 << 20):.2f} | {r['engine_size_ratio_vs_fp32']:.2f} |")
    (root / "report.md").write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--models", default="resnet18,resnet50")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--checkpoint")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--out")
    parser.add_argument("--out-dir", default="results/quantization")
    parser.add_argument("--engine-dir", default="engines")
    parser.add_argument("--calibration-dir")
    parser.add_argument("--eval-dir")
    parser.add_argument("--dataset-label", default="ImageNet validation")
    parser.add_argument("--class-index", help="JSON mapping folder name -> original ImageNet class index")
    parser.add_argument("--calibration-samples", type=positive_int, default=512)
    parser.add_argument("--eval-samples", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--workspace-gib", type=positive_int, default=2)
    args = parser.parse_args()
    if args.suite:
        if not args.calibration_dir or not args.eval_dir:
            parser.error("--suite requires --calibration-dir and --eval-dir")
        suite(args)
    else:
        if not args.checkpoint or not args.out:
            parser.error("single-engine export requires --checkpoint and --out")
        require_cuda()
        source = Path(args.out).with_suffix('.onnx')
        export_to_onnx(args.checkpoint, source, args.batch_size, args.model)
        if args.precision == "int8":
            if not args.calibration_dir:
                parser.error("INT8 requires --calibration-dir")
            target = source.with_suffix('.qdq.onnx')
            data = subset(image_dataset(args.calibration_dir, args.model, labels_required=False), args.calibration_samples, args.seed)
            quantize_onnx(source, target, data, args.batch_size)
            source = target
        build_engine(source, args.out, args.precision, args.workspace_gib)


if __name__ == "__main__":
    main()
