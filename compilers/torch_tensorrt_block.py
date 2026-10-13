"""Compare identical LLaMA-style blocks/ResNets with eager, Inductor, Torch-TRT and native TRT.

The LLaMA block has RMSNorm, RoPE, causal attention and SwiGLU; seeded random
weights exercise compilation, not language quality. No KV cache/backward.
"""
import argparse
import gc
import os
import time
from pathlib import Path

from harness.measurement import metadata, positive_int, synchronized_time, write_json
from harness.tensorrt_utils import TensorRTRunner, build_engine, require_cuda
from training.export_tensorrt import export_model, load_model


def llama_block(hidden, heads, sequence, intermediate):
    import torch
    from torch import nn
    if hidden % heads or (hidden // heads) % 2:
        raise ValueError("hidden must divide heads with an even head dimension")
    dim = hidden // heads

    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden))
        def forward(self, x):
            values = x.float()
            normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            return normed.to(x.dtype) * self.weight

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = RMSNorm(), RMSNorm()
            self.q, self.k, self.v, self.o = [nn.Linear(hidden, hidden, bias=False) for _ in range(4)]
            self.gate, self.up = [nn.Linear(hidden, intermediate, bias=False) for _ in range(2)]
            self.down = nn.Linear(intermediate, hidden, bias=False)
            freq = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            angles = torch.arange(sequence).float()[:, None] * freq[None, :]
            angles = torch.cat([angles, angles], dim=-1)[None, None]
            self.register_buffer("cos", angles.cos())
            self.register_buffer("sin", angles.sin())
            mask = torch.full((sequence, sequence), float('-inf')).triu(1)
            self.register_buffer("mask", mask)
        def rotate(self, x):
            a, b = x.chunk(2, dim=-1)
            return x * self.cos + torch.cat([-b, a], dim=-1) * self.sin
        def forward(self, x):
            n = self.norm1(x)
            q, k, v = [layer(n).reshape(x.shape[0], sequence, heads, dim).transpose(1, 2)
                       for layer in (self.q, self.k, self.v)]
            q, k = self.rotate(q), self.rotate(k)
            scores = (q @ k.transpose(-1, -2)).float() / dim ** .5 + self.mask
            probabilities = scores.softmax(-1).to(v.dtype)
            attended = (probabilities @ v).transpose(1, 2).reshape(x.shape[0], sequence, hidden)
            x = x + self.o(attended)
            n = self.norm2(x)
            return x + self.down(torch.nn.functional.silu(self.gate(n)) * self.up(n))
    return Block().eval()


def explain_graph(model, x):
    import torch
    explanation = torch._dynamo.explain(model)(x)
    return {"graph_count": explanation.graph_count, "graph_break_count": explanation.graph_break_count,
            "op_count": explanation.op_count,
            "break_reasons": [{"reason": r.reason, "user_stack": [str(f) for f in r.user_stack]}
                              for r in explanation.break_reasons]}


def compare(actual, expected, precision):
    import torch
    atol, rtol = (3e-2, 3e-2) if precision == 'fp16' else (1e-4, 1e-4)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {"status": "pass", "atol": atol, "rtol": rtol,
            "max_abs_error": (actual.float() - expected.float()).abs().max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='llama-block,resnet18,resnet50')
    parser.add_argument('--batch-size', type=positive_int, default=1)
    parser.add_argument('--sequence', type=positive_int, default=128)
    parser.add_argument('--hidden', type=positive_int, default=512)
    parser.add_argument('--heads', type=positive_int, default=8)
    parser.add_argument('--intermediate', type=positive_int, default=1376)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--iterations', type=positive_int, default=50)
    parser.add_argument('--warmup', type=positive_int, default=10)
    parser.add_argument('--out-dir', default='results/compilers/torch-tensorrt')
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_LOGS', 'graph_breaks')
    torch = require_cuda()
    import torch_tensorrt
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    result = {'metadata': metadata(['torch', 'torchvision', 'torch_tensorrt', 'tensorrt', 'onnx']),
              'config': vars(args), 'llama_scope': 'seeded random LLaMA-style block; no pretrained checkpoint or KV cache',
              'models': {}}
    failures = []
    for name in args.models.split(','):
        if name == 'llama-block':
            model = llama_block(args.hidden, args.heads, args.sequence, args.intermediate)
            shape = (args.batch_size, args.sequence, args.hidden)
        elif name in ('resnet18', 'resnet50'):
            model = load_model(name)
            shape = (args.batch_size, 3, 224, 224)
        else:
            parser.error(f'Unsupported model {name}')
        model = model.cuda().to(dtype)
        x = torch.randn(shape, device='cuda', dtype=dtype)
        report = {'shape': shape, 'backends': {}}
        result['models'][name] = report
        with torch.inference_mode():
            expected = model(x).clone()
            report['dynamo_explain'] = explain_graph(model, x)
            write_json(root / f'{name}-graph-breaks.json', report['dynamo_explain'])
            for backend in ('eager', 'torch.compile', 'torch-tensorrt', 'native-tensorrt'):
                fn = compiled = native = program = None
                try:
                    torch._dynamo.reset()
                    torch._dynamo.utils.counters.clear()
                    start = time.perf_counter()
                    details = {}
                    if backend == 'eager':
                        fn = lambda: model(x)
                    elif backend == 'torch.compile':
                        compiled = torch.compile(model, backend='inductor', fullgraph=False)
                        fn = lambda: compiled(x)
                    elif backend == 'torch-tensorrt':
                        program = torch.export.export(model, (x,))
                        compiled = torch_tensorrt.dynamo.compile(
                            program, inputs=[x], enabled_precisions={dtype}, min_block_size=1,
                            require_full_compilation=False, disable_tf32=True)
                        modules = [{'name': n, 'type': type(m).__module__ + '.' + type(m).__name__}
                                   for n, m in compiled.named_modules() if n]
                        engines = [m for m in modules if 'tensorrt' in m['type'].lower()]
                        if not engines:
                            raise RuntimeError('Torch-TensorRT produced no TensorRT runtime modules')
                        (root / f'{name}-partitioned-graph.txt').write_text(str(compiled.graph))
                        details = {'partition_modules': modules, 'trt_runtime_modules': engines,
                                   'partitioned_graph': str(root / f'{name}-partitioned-graph.txt'),
                                   'note': 'PyTorch fallback partitions are separate from Dynamo graph breaks'}
                        def fn():
                            value = compiled(x)
                            return value[0] if isinstance(value, (tuple, list)) else value
                    else:
                        onnx_path, engine_path = root / f'{name}.onnx', root / f'{name}.trt'
                        export_model(model, x, onnx_path)
                        build_engine(onnx_path, engine_path, args.precision)
                        native = TensorRTRunner(engine_path)
                        fn = lambda: native(input=x)['output']
                        write_json(root / f'{name}-native-layers.json', native.inspect())
                        details = {'engine_size_bytes': engine_path.stat().st_size, 'engine': str(engine_path)}
                    actual = fn()
                    torch.cuda.synchronize()
                    cold_ms = (time.perf_counter() - start) * 1000
                    correct = compare(actual, expected, args.precision)
                    timing = synchronized_time(fn, args.warmup, args.iterations)
                    counters = {str(k): dict(v) for k, v in torch._dynamo.utils.counters.items() if v}
                    report['backends'][backend] = {'status': 'pass', 'correctness': correct, 'latency': timing,
                                                   'build_and_first_call_ms': cold_ms,
                                                   'dynamo_counters': counters, **details}
                    print(f'{name} {backend}: {timing["median_ms"]:.3f} ms')
                except Exception as error:
                    report['backends'][backend] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
                    failures.append(f'{name}/{backend}')
                    print(f'{name} {backend} failed: {error}')
                finally:
                    write_json(root / 'benchmark.json', result)
                    fn = compiled = native = program = None
                    gc.collect()
                    torch.cuda.empty_cache()
        eager = report['backends']['eager']
        if eager['status'] == 'pass':
            for row in report['backends'].values():
                if row['status'] == 'pass':
                    row['speedup_vs_eager'] = eager['latency']['median_ms'] / row['latency']['median_ms']
        del model, x, expected
    result['failures'] = failures
    write_json(root / 'benchmark.json', result)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
