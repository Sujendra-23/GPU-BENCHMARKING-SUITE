"""Whisper-tiny English ASR: identical fixed-shape encoder/decoder across runtimes.

Greedy decoding recomputes a padded decoder sequence without a KV cache.
This is an end-to-end transcription benchmark, not a production ASR decoder.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from harness.measurement import metadata, synchronized_time, write_json
from harness.onnx_utils import ONNXRunner
from harness.tensorrt_utils import TensorRTRunner, build_engine, require_cuda
from training.export_tensorrt import export_model

MODEL = 'openai/whisper-tiny'


def normalized_words(text):
    return re.sub(r"[^\w\s']", ' ', text.lower()).split()


def word_errors(reference, hypothesis):
    a, b = normalized_words(reference), normalized_words(hypothesis)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1], len(a)


def load_audio_manifest(path, limit):
    import numpy as np
    import soundfile as sf
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio_path = Path(row['audio'])
        if not audio_path.is_absolute():
            audio_path = Path(path).resolve().parent / audio_path
        audio, rate = sf.read(audio_path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 16000:
            from scipy.signal import resample_poly
            import math
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        if not 0 < len(audio) <= 30 * 16000 or not np.isfinite(audio).all():
            raise ValueError('Whisper benchmark requires finite nonempty audio clips <= 30 seconds')
        if not isinstance(row.get('text'), str) or not normalized_words(row['text']):
            raise ValueError('Every audio clip needs a nonempty reference transcript')
        rows.append({'audio': audio, 'path': str(audio_path), 'text': row['text'],
                     'seconds': len(audio) / 16000, 'sha256': hashlib.sha256(audio_path.read_bytes()).hexdigest()})
        if len(rows) == limit:
            break
    if not rows:
        raise ValueError('Audio manifest is empty')
    return rows


def components(model, width):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.model.encoder
        def forward(self, features):
            return self.encoder(features, return_dict=False)[0]

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder, self.proj = model.model.decoder, model.proj_out
            # A supplied 4-D mask bypasses Transformers' vmap mask construction,
            # which the legacy ONNX tracer cannot export. Shape is deliberately fixed.
            mask = torch.full((1, 1, width, width), float('-inf'), device=model.device, dtype=model.dtype).triu(1)
            self.register_buffer('causal_mask', mask)
        def forward(self, input_ids, hidden, position):
            states = self.decoder(input_ids=input_ids, encoder_hidden_states=hidden,
                                  attention_mask=self.causal_mask, use_cache=False, return_dict=False)[0]
            selected = torch.index_select(states, 1, position).squeeze(1)
            return self.proj(selected)
    return Encoder().eval(), Decoder().eval()


class ASRRunner:
    def __init__(self, backend, model, processor, batch, precision, max_tokens, root):
        torch = require_cuda()
        self.torch, self.processor, self.batch = torch, processor, batch
        self.dtype = torch.float16 if precision == 'fp16' else torch.float32
        self.max_tokens = max_tokens
        forced = processor.get_decoder_prompt_ids(language='english', task='transcribe', no_timestamps=True)
        self.prefix = [model.config.decoder_start_token_id] + [token for _, token in forced]
        self.width = len(self.prefix) + max_tokens
        if self.width > model.config.max_target_positions:
            raise ValueError('Decoder prompt plus generated tokens exceeds Whisper positional capacity')
        self.eos = model.config.eos_token_id
        self.suppressed = model.generation_config.suppress_tokens or []
        self.begin_suppressed = model.generation_config.begin_suppress_tokens or []
        encoder, decoder = components(model, self.width)
        self.features = torch.zeros(batch, 80, 3000, device='cuda', dtype=self.dtype)
        self.ids = torch.full((batch, self.width), self.eos, device='cuda', dtype=torch.int64)
        self.position = torch.tensor([len(self.prefix) - 1], device='cuda', dtype=torch.int64)
        sample_hidden = torch.zeros(batch, 1500, model.config.d_model, device='cuda', dtype=self.dtype)
        start = time.perf_counter()
        if backend in ('torch', 'torch.compile'):
            if backend == 'torch.compile':
                encoder = torch.compile(encoder, fullgraph=False)
                decoder = torch.compile(decoder, fullgraph=False)
            self.encode = lambda features: encoder(features)
            self.decode = lambda ids, hidden, position: decoder(ids, hidden, position)
        else:
            base = Path(root) / f'bs{batch}-{precision}'
            base.mkdir(parents=True, exist_ok=True)
            enc_path, dec_path = base / 'encoder.onnx', base / 'decoder.onnx'
            export_model(encoder, self.features, enc_path, ('features',), ('hidden',))
            export_model(decoder, (self.ids, sample_hidden, self.position), dec_path,
                         ('input_ids', 'hidden', 'position'), ('logits',))
            if backend == 'onnx':
                enc = ONNXRunner(enc_path, {'hidden': (batch, 1500, model.config.d_model)})
                dec = ONNXRunner(dec_path, {'logits': (batch, model.config.vocab_size)})
            elif backend == 'tensorrt':
                enc_engine, dec_engine = base / 'encoder.trt', base / 'decoder.trt'
                build_engine(enc_path, enc_engine, precision)
                build_engine(dec_path, dec_engine, precision)
                enc, dec = TensorRTRunner(enc_engine), TensorRTRunner(dec_engine)
            else:
                raise ValueError(f'Unsupported ASR backend: {backend}')
            self.encode = lambda features: enc(features=features)['hidden']
            self.decode = lambda ids, hidden, position: dec(input_ids=ids, hidden=hidden, position=position)['logits']
        # Exercise both graphs before reporting steady-state measurements.
        with torch.inference_mode():
            hidden = self.encode(self.features)
            self.decode(self.ids, hidden, self.position)
            torch.cuda.synchronize()
        self.build_first_call_ms = (time.perf_counter() - start) * 1000

    def transcribe(self, features):
        torch = self.torch
        with torch.inference_mode():
            self.ids.fill_(self.eos)
            self.ids[:, :len(self.prefix)] = torch.tensor(self.prefix, device='cuda')
            hidden = self.encode(features)
            done = torch.zeros(self.batch, device='cuda', dtype=torch.bool)
            lengths = torch.full((self.batch,), self.max_tokens, device='cuda', dtype=torch.int64)
            for step in range(self.max_tokens):
                index = len(self.prefix) - 1 + step
                self.position.fill_(index)
                logits = self.decode(self.ids, hidden, self.position)
                # Masking is shared by all runtimes; NaN/Inf outputs are checked before timing.
                if self.suppressed:
                    logits[:, self.suppressed] = float('-inf')
                if step == 0 and self.begin_suppressed:
                    logits[:, self.begin_suppressed] = float('-inf')
                token = logits.argmax(-1)
                token = torch.where(done, self.eos, token)
                newly_done = ~done & (token == self.eos)
                lengths = torch.where(newly_done, step + 1, lengths)
                done |= newly_done
                self.ids[:, index + 1] = token
                if done.all().item():
                    break
            return self.ids[:, len(self.prefix):].clone(), lengths


def benchmark(args):
    torch = require_cuda()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from harness.benchmark import BenchResult, query_gpu_metrics, write_csv
    if not args.audio_manifest:
        raise ValueError('--model whisper-tiny requires --audio-manifest JSONL with audio and text')
    if any(p not in ('fp32', 'fp16') for p in args.precisions.split(',')):
        raise ValueError('Whisper currently supports FP32/FP16')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = WhisperProcessor.from_pretrained(MODEL, revision=args.model_revision)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=args.model_revision,
                                                            attn_implementation='eager').eval().cuda()
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    clips = load_audio_manifest(args.audio_manifest, args.audio_samples)
    report = {'metadata': metadata(['torch', 'transformers', 'tensorrt', 'onnxruntime', 'soundfile']),
              'config': vars(args), 'resolved_model_commit': getattr(model.config, '_commit_hash', None),
              'method': 'English greedy ASR; fixed 30-second mel input; padded decoder, no KV cache; CPU preprocessing excluded from timed region',
              'wer_normalization': 'lowercase, replace punctuation except apostrophes with spaces, word edit distance',
              'clips': [{k: v for k, v in c.items() if k != 'audio'} for c in clips], 'results': []}
    rows = []
    root = Path(args.asr_out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for batch in [int(b) for b in args.batch_sizes.split(',')]:
        for precision in args.precisions.split(','):
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            model.float().load_state_dict(original_state)
            model.to(dtype)
            features = []
            groups = []
            for offset in range(0, len(clips), batch):
                group = clips[offset:offset + batch]
                arrays = [c['audio'] for c in group]
                arrays += [arrays[-1]] * (batch - len(arrays))
                mel = processor(arrays, sampling_rate=16000, return_tensors='pt').input_features.to('cuda', dtype=dtype)
                features.append(mel.contiguous())
                groups.append(group)
            reference = ASRRunner('torch', model, processor, batch, precision, args.max_new_tokens, root / 'engines')
            with torch.inference_mode():
                reference_hidden = reference.encode(features[0]).clone()
                probe_ids = reference.ids.clone()
                probe_ids[:, :len(reference.prefix)] = torch.tensor(reference.prefix, device='cuda')
                probe_position = reference.position.clone()
                reference_logits = reference.decode(probe_ids, reference_hidden, probe_position).clone()
            for backend in args.frameworks.split(','):
                runner = reference if backend == 'torch' else ASRRunner(backend, model, processor, batch, precision, args.max_new_tokens, root / 'engines')
                with torch.inference_mode():
                    actual_hidden = runner.encode(features[0]).clone()
                    actual_logits = runner.decode(probe_ids, reference_hidden, probe_position).clone()
                    tolerance = 3e-2 if precision == 'fp16' else 1e-3
                    torch.testing.assert_close(actual_hidden, reference_hidden, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(actual_logits, reference_logits, atol=tolerance, rtol=tolerance)
                errors = words = 0
                transcripts = []
                for mel, group in zip(features, groups):
                    tokens, lengths = runner.transcribe(mel)
                    texts = processor.batch_decode(tokens.cpu(), skip_special_tokens=True)
                    for i, clip in enumerate(group):
                        count, total = word_errors(clip['text'], texts[i])
                        errors += count
                        words += total
                        transcripts.append({'audio': clip['path'], 'reference': clip['text'], 'hypothesis': texts[i],
                                            'generated_tokens': lengths[i].item(),
                                            'hit_token_limit': lengths[i].item() == args.max_new_tokens and tokens[i, -1].item() != runner.eos})
                # Cycle through actual clips during timing; includes greedy-loop CPU/GPU synchronization.
                index = 0
                timed_groups = []
                def call():
                    nonlocal index
                    selected = index % len(features)
                    runner.transcribe(features[selected])
                    timed_groups.append(selected)
                    index += 1
                timing = synchronized_time(call, args.warmup, args.iterations)
                util, power = query_gpu_metrics()
                measured_groups = timed_groups[args.warmup:]
                valid_per_batch = sum(len(groups[i]) for i in measured_groups) / len(measured_groups)
                audio_seconds_per_batch = sum(sum(c['seconds'] for c in groups[i]) for i in measured_groups) / len(measured_groups)
                row = {'framework': backend, 'batch_size': batch, 'precision': precision,
                       'build_and_first_call_ms': runner.build_first_call_ms, 'latency': timing,
                       'wer': errors / words, 'word_errors': errors, 'reference_words': words,
                       'rtf_on_mean_batch_duration': timing['mean_ms'] / (1000 * audio_seconds_per_batch),
                       'timed_batches_cycle': len(features), 'mean_valid_clips_per_timed_batch': valid_per_batch,
                       'transcripts': transcripts,
                       'correctness': {'status': 'pass', 'atol': tolerance, 'rtol': tolerance,
                                       'encoder_max_abs_error': (actual_hidden - reference_hidden).abs().max().item(),
                                       'decoder_first_step_max_abs_error': (actual_logits - reference_logits).abs().max().item()}}
                report['results'].append(row)
                rows.append(BenchResult(backend, batch, precision, round(timing['mean_ms'], 4),
                                        round(timing['p95_ms'], 4), round(valid_per_batch * 1000 / timing['mean_ms'], 2),
                                        util, power, 'whisper-tiny'))
                write_json(root / 'benchmark.json', report)
                print(f"Whisper {backend} {precision} bs={batch}: WER={row['wer']:.3f}, {timing['mean_ms']:.2f} ms")
                del runner
            del reference, features
    write_csv(rows, args.out)
