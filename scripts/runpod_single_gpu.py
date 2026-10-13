"""Run the prepared single-GPU workloads sequentially; preserve logs and failures.

Install requirements-gpu-extensions.txt and vLLM first. TensorRT-LLM runs in
its own environment using serving.run, because its dependency pins conflict.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from harness.measurement import metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/imagenette2-160')
    parser.add_argument('--audio-manifest', default='data/whisper/manifest.jsonl')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--stages', default='quantization,serving,compiler,edge,whisper')
    parser.add_argument('--out-dir', default='results/runpod')
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    engine_dir = str(root / 'engines')
    quick = args.quick
    commands = {
        'quantization': ['training.export_tensorrt', '--suite', '--batch-size', '1',
                         '--calibration-dir', str(Path(args.data_dir) / 'train'),
                         '--eval-dir', str(Path(args.data_dir) / 'val'),
                         '--class-index', 'training/imagenette-class-index.json',
                         '--dataset-label', 'Imagenette-160 held-out validation; 10-class subset, original 1000 logits',
                         '--calibration-samples', '64' if quick else '512',
                         '--eval-samples', '128' if quick else '1000',
                         '--engine-dir', engine_dir, '--out-dir', str(root / 'quantization')],
        'serving': ['serving.run', '--backend', 'vllm', '--quantization', 'fp8',
                    '--requests', '16' if quick else '64', '--concurrency', '1,4,8,16',
                    '--out-dir', str(root / 'serving')],
        'compiler': ['compilers.torch_tensorrt_block', '--models', 'llama-block,resnet18' if quick else 'llama-block,resnet18,resnet50', '--batch-size', '1',
                     '--iterations', '20' if quick else '50', '--out-dir', str(root / 'compiler')],
        'edge': ['harness.edge_profile', '--power-limit', '200', '--model', 'resnet18',
                 '--engine-dir', engine_dir, '--out-dir', str(root / 'edge')],
        'whisper': ['harness.benchmark', '--model', 'whisper-tiny', '--batch-sizes', '1',
                    '--frameworks', 'torch,onnx,tensorrt,torch.compile', '--precisions', 'fp32',
                    '--audio-manifest', args.audio_manifest, '--audio-samples', '4' if quick else '8',
                    '--warmup', '1' if quick else '3', '--iterations', '3' if quick else '20',
                    '--out', str(root / 'whisper.csv'), '--asr-out-dir', str(root / 'whisper')],
    }
    import json
    existing = json.loads((root / 'status.json').read_text()).get('stages', {}) if (root / 'status.json').exists() else {}
    report = {'metadata': metadata(['torch', 'torch_tensorrt', 'vllm']), 'config': vars(args), 'stages': existing}
    for stage in args.stages.split(','):
        if stage not in commands:
            parser.error(f'Unknown stage {stage}')
        # A paired FP16 vLLM baseline is required for the quantized comparison.
        work = [commands[stage]]
        if stage == 'serving':
            baseline = commands[stage].copy()
            baseline[baseline.index('--quantization') + 1] = 'none'
            work.insert(0, baseline)
        outcomes = []
        for i, arguments in enumerate(work):
            command = [sys.executable, '-u', '-m', *arguments]
            log_path = root / f'{stage}-{i}.log'
            print(f'Running {stage}: {command}', flush=True)
            with log_path.open('w') as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            outcomes.append({'command': command, 'returncode': status, 'log': str(log_path)})
            report['stages'][stage] = outcomes
            write_json(root / 'status.json', report)
    raise SystemExit(1 if any(r['returncode'] for rows in report['stages'].values() for r in rows) else 0)


if __name__ == '__main__':
    main()
