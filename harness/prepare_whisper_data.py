"""Save a small public LibriSpeech fixture as local WAVs plus reference JSONL."""
import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='data/whisper')
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    source = hf_hub_download('hf-internal-testing/librispeech_asr_dummy',
                            'clean/validation/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(source).to_pylist()[:args.samples]
    with (root / 'manifest.jsonl').open('w') as manifest:
        for i, row in enumerate(rows):
            audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
            name = f'{i:03d}.wav'
            sf.write(root / name, audio, rate)
            manifest.write(json.dumps({'audio': name, 'text': row['text']}) + '\n')
    (root / 'source.json').write_text(json.dumps({'dataset': 'hf-internal-testing/librispeech_asr_dummy',
                                                'source_snapshot': source, 'samples': len(rows)}, indent=2) + '\n')
    print(root / 'manifest.jsonl')


if __name__ == '__main__':
    main()
