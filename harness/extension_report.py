"""Render only persisted GPU measurements and explicit pending/failure states."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def render(root):
    lines = ['# Additional GPU experiments', '', 'Measured data only. Missing results remain pending.', '']
    quant = read(root / 'quantization/benchmark.json')
    lines += ['## TensorRT quantization', '']
    if quant:
        lines += [f"Dataset: {quant.get('dataset_label', 'see configuration')}. Calibration/evaluation sizes are recorded in JSON.", '',
                  '| model | precision | top-1 % | delta pp | median ms | speedup vs FP32 | MiB |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for model, report in quant['models'].items():
            for row in report['results']:
                lines.append(f"| {model} | {row['precision']} | {row['accuracy']['top1_percent']:.2f} | {row['top1_delta_pp_vs_fp32']:.2f} | {row['latency']['median_ms']:.3f} | {row['latency_speedup_vs_fp32']:.2f} | {row['engine_size_bytes'] / (1 << 20):.2f} |")
    else:
        lines += ['Pending GPU execution.']
    lines += ['', '## Serving', '', '| runtime/run | concurrency | TTFT p50 ms | output tokens/s | failed requests |',
              '| --- | ---: | ---: | ---: | ---: |']
    found = False
    for path in sorted((root / 'serving').glob('*.json')):
        report = read(path)
        if not isinstance(report, dict) or 'results' not in report:
            continue
        found = True
        for row in report['results']:
            ttft = row['ttft_p50_ms']
            value = f'{ttft:.2f}' if ttft is not None else 'unavailable'
            lines.append(f"| {path.stem} | {row['concurrency']} | {value} | {row['output_tokens_per_second']:.2f} | {row['failed']} |")
    if not found:
        lines += ['', 'Pending GPU execution.']
    lines += ['', 'Synthetic token prompts measure serving performance, not language quality.', '', '## Compilers', '']
    compiler = read(root / 'compiler/benchmark.json')
    if compiler:
        lines += ['| model | backend | status | median ms | speedup vs eager |', '| --- | --- | --- | ---: | ---: |']
        break_summaries = []
        for model, report in compiler['models'].items():
            for backend, row in report['backends'].items():
                latency = f"{row['latency']['median_ms']:.3f}" if row['status'] == 'pass' else '—'
                speedup = f"{row['speedup_vs_eager']:.2f}" if 'speedup_vs_eager' in row else '—'
                lines.append(f"| {model} | {backend} | {row['status']} | {latency} | {speedup} |")
            explain = report.get('dynamo_explain', {})
            break_summaries.append(f"{model}: {explain.get('graph_break_count', 'unavailable')}")
        lines += ['', 'Dynamo graph breaks: ' + '; '.join(break_summaries) + '.', '',
                  'LLaMA-style block uses seeded random weights; compiler correctness is not model accuracy.']
    else:
        lines += ['Pending GPU execution.']
    lines += ['', '## Edge', '']
    edge = read(root / 'edge/profile.json')
    if edge:
        lines += [f"Label: **{edge['label']}**. Cap status: `{edge['cap_status']}`; requested {edge['requested_limit_watts']} W.",
                  f"Original limit: {edge['original']['current_watts']} W. Restore verified: {edge.get('restore_verified', 'no cap attempted')}."]
    else:
        lines += ['Pending GPU execution.']
    lines += ['', '## Whisper-tiny', '']
    whisper = read(root / 'whisper/benchmark.json')
    if whisper:
        lines += ['| backend | batch | precision | WER | mean ms | RTF |', '| --- | ---: | --- | ---: | ---: | ---: |']
        for row in whisper['results']:
            lines.append(f"| {row['framework']} | {row['batch_size']} | {row['precision']} | {row['wer']:.3f} | {row['latency']['mean_ms']:.2f} | {row['rtf_on_mean_batch_duration']:.3f} |")
        lines += ['', whisper['method'], 'The LibriSpeech dummy fixture is a functional smoke test, not a representative ASR evaluation.']
    else:
        lines += ['Pending GPU execution.']
    lines += ['', '## Extra rentals', '']
    nsight = read(root / 'nsight-probe/status.json')
    lines += [f"Nsight probe: `{nsight['status']}`." if nsight else 'Nsight counter probe pending.',
              'TP2 needs two visible GPUs in one pod; compare TP1 and TP2 on that same host.', '']
    pending = read(root / 'external-pending.json')
    if pending:
        lines += ['', '## Prepared but not executed', '']
        for name, row in pending.items():
            lines.append(f"- {name}: {row['reason']}")
    status = read(root / 'status.json')
    if status:
        lines += ['', '## Stage exit codes', '']
        for name, rows in status['stages'].items():
            lines.append(f"- {name}: {', '.join(str(row['returncode']) for row in rows)}; see stage logs.")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/runpod'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    path = args.out or args.root / 'report.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(args.root))
    print(path)


if __name__ == '__main__':
    main()
