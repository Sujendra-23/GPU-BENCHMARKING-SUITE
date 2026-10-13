# Additional GPU experiments

Measured data only. Missing results remain pending.

## TensorRT quantization

Dataset: Imagenette-160 held-out validation; 10-class subset, original 1000 logits. Calibration/evaluation sizes are recorded in JSON.

| model | precision | top-1 % | delta pp | median ms | speedup vs FP32 | MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| resnet18 | fp32 | 59.38 | 0.00 | 0.504 | 1.00 | 73.79 |
| resnet18 | fp16 | 59.38 | 0.00 | 0.190 | 2.65 | 23.47 |
| resnet18 | int8 | 60.94 | 1.56 | 0.186 | 2.72 | 12.57 |
| resnet50 | fp32 | 82.03 | 0.00 | 0.991 | 1.00 | 124.36 |
| resnet50 | fp16 | 82.03 | 0.00 | 0.353 | 2.81 | 50.27 |
| resnet50 | int8 | 81.25 | -0.78 | 0.364 | 2.72 | 27.17 |

## Serving

| runtime/run | concurrency | TTFT p50 ms | output tokens/s | failed requests |
| --- | ---: | ---: | ---: | ---: |
| vllm-fp8-tp1 | 1 | 16.79 | 77.16 | 0 |
| vllm-fp8-tp1 | 4 | 26.25 | 289.14 | 0 |
| vllm-fp8-tp1 | 8 | 38.19 | 563.95 | 0 |
| vllm-fp8-tp1 | 16 | 58.49 | 1097.79 | 0 |
| vllm-none-tp1 | 1 | 13.41 | 96.22 | 0 |
| vllm-none-tp1 | 4 | 22.32 | 350.19 | 0 |
| vllm-none-tp1 | 8 | 32.25 | 679.57 | 0 |
| vllm-none-tp1 | 16 | 69.52 | 1260.27 | 0 |

Synthetic token prompts measure serving performance, not language quality.

## Compilers

| model | backend | status | median ms | speedup vs eager |
| --- | --- | --- | ---: | ---: |
| llama-block | eager | pass | 0.757 | 1.00 |
| llama-block | torch.compile | pass | 0.374 | 2.03 |
| llama-block | torch-tensorrt | pass | 0.176 | 4.30 |
| llama-block | native-tensorrt | pass | 0.070 | 10.84 |
| resnet18 | eager | pass | 2.309 | 1.00 |
| resnet18 | torch.compile | pass | 1.301 | 1.77 |
| resnet18 | torch-tensorrt | pass | 0.281 | 8.22 |
| resnet18 | native-tensorrt | pass | 0.189 | 12.22 |

Dynamo graph breaks: llama-block: 0; resnet18: 0.

LLaMA-style block uses seeded random weights; compiler correctness is not model accuracy.

## Edge

Label: **edge-style, not Jetson**. Cap status: `refused_or_unverified`; requested 200 W.
Original limit: 450.0 W. Restore verified: True.

## Whisper-tiny

| backend | batch | precision | WER | mean ms | RTF |
| --- | ---: | --- | ---: | ---: | ---: |
| torch | 1 | fp32 | 0.060 | 115.04 | 0.013 |
| onnx | 1 | fp32 | 0.060 | 51.45 | 0.006 |
| tensorrt | 1 | fp32 | 0.060 | 26.59 | 0.003 |
| torch.compile | 1 | fp32 | 0.060 | 67.39 | 0.007 |

English greedy ASR; fixed 30-second mel input; padded decoder, no KV cache; CPU preprocessing excluded from timed region
The LibriSpeech dummy fixture is a functional smoke test, not a representative ASR evaluation.

## Extra rentals

Nsight probe: `permission_denied`.
TP2 needs two visible GPUs in one pod; compare TP1 and TP2 on that same host.


## Follow-up status

- tensorrt-llm: `executed_on_separate_host`. TensorRT-LLM TP1 completed on two H100s, with zero failed requests and valid KV metrics. See [the H100 report](../h100-dual/report.md).
- vllm-tp2: `executed_on_separate_host`. Matched vLLM TP1 and TP2 completed on two H100s, with zero failed requests and valid KV metrics. See [the H100 report](../h100-dual/report.md).
- nsight-counters: `permission_denied`. Nsight Compute returned ERR_NVGPUCTRPERM on the RTX 4090 and on the H100 host before and after restart. Counter collection still needs provider-enabled access; no hardware counter values are claimed.

## Stage exit codes

- quantization: 0; see stage logs.
- serving: 0, 0; see stage logs.
- compiler: 0; see stage logs.
- edge: 0; see stage logs.
- whisper: 0; see stage logs.
