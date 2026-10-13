# Two-H100 serving results

All three serving phases completed successfully on **two H100 80 GB HBM3 GPUs**, driver 570.124.06, with `NV18` between GPU0 and GPU1. TensorRT-LLM and vLLM TP1 used GPU0; vLLM TP2 used GPU0 and GPU1. Topology is retained in `topology.txt`.

Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`; FP16 weights and KV; 32 requests at each concurrency; exact 256 input / 128 output tokens per request; two warmup requests before the sweep. Closed-loop client timing, graphs and prefix reuse disabled, 2,048-token context and scheduled-token limits, maximum 16 sequences. The total configured KV payload capacity is 65,536 tokens; TP2 shards it across two ranks. These are quick single-run performance measurements, not language-quality results.

| Runtime | TP | Concurrency | TTFT p50 ms | TTFT p95 ms | Output tokens/s | KV peak fraction (sampled) | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorRT-LLM | 1 | 1 | 31.19 | 42.89 | 73.28 | 0.00586 | 0 |
| TensorRT-LLM | 1 | 4 | 46.15 | 69.62 | 270.88 | 0.02344 | 0 |
| TensorRT-LLM | 1 | 8 | 51.10 | 71.98 | 539.53 | 0.04688 | 0 |
| TensorRT-LLM | 1 | 16 | 86.95 | 129.52 | 885.96 | 0.09375 | 0 |
| vLLM | 1 | 1 | 10.32 | 14.06 | 144.83 | 0.00586 | 0 |
| vLLM | 1 | 4 | 15.03 | 37.12 | 553.30 | 0.02344 | 0 |
| vLLM | 1 | 8 | 20.60 | 37.48 | 1127.67 | 0.04689 | 0 |
| vLLM | 1 | 16 | 35.58 | 62.31 | 2136.35 | 0.09377 | 0 |
| vLLM | 2 | 1 | 12.37 | 13.86 | 113.96 | 0.00586 | 0 |
| vLLM | 2 | 4 | 18.13 | 38.36 | 458.20 | 0.02344 | 0 |
| vLLM | 2 | 8 | 27.15 | 43.03 | 903.05 | 0.04689 | 0 |
| vLLM | 2 | 16 | 34.25 | 60.09 | 1731.26 | 0.09377 | 0 |

All 384 measured requests succeeded with the requested token counts; every load level had a valid KV measurement and no metrics errors. `status.json` records exit code 0 for TensorRT-LLM, TP1 and TP2. Raw request results, sampled counters, launch commands and server logs are under `serving/`.

## Matched vLLM TP1 versus TP2

| Concurrency | TP2 / TP1 throughput | Throughput change |
| ---: | ---: | ---: |
| 1 | 0.787× | -21.3% |
| 4 | 0.828× | -17.2% |
| 8 | 0.801× | -19.9% |
| 16 | 0.810× | -19.0% |

TP2 was slower for this small model and workload despite NVLink. At concurrency 16, TP1 delivered 2,136.35 output tokens/s and TP2 1,731.26. This result does not predict scaling for larger models. KV fractions retain engine labels; estimated used bytes are per rank, not aggregate GPU memory.

## TensorRT-LLM executor and setup

TensorRT-LLM **1.0.0 uses its PyTorch executor**, not a serialized TensorRT-LLM engine. Its isolated Python 3.10 environment uses PyTorch 2.7.1+cu128 and TensorRT 10.11.0.33. vLLM 0.10.2 uses isolated Python 3.11 with PyTorch 2.8.0+cu128. Both dependency checks passed; exact versions are in the environment manifests.

The launcher adapts only TensorRT-LLM 1.0.0’s HTTP metrics reader, draining real iteration stats for at most 20 ms and retaining collected stats on its empty-deque error. Model execution and the stats producer are unchanged. The original endpoint timed out during sustained load; that first measured attempt, including its partial/failed metrics, is preserved under `trtllm-original-metrics/` and is excluded from the table above. The accepted run has zero metrics errors.

Python 3.11 had no NVIDIA 1.0.0 wheel. The initial CUDA 12.6 PyTorch build also failed at DeepGEMM import (`cudaLibraryGetKernel`); explicit CUDA 12.8 wheel pins fixed it. FlashInfer JIT required Ninja in the environment’s PATH. Setup and failed-start logs are retained, and the corrected commands are in [docs/runpod.md](../../docs/runpod.md).

## GPU recovery and remaining counters

Before these measurements, the container lost GPU device access: `nvidia-smi` returned `Failed to initialize NVML: Unknown Error`, and device opens returned `EPERM`. The user’s RunPod restart restored access and preserved the installed environment. Failure and recovery logs are retained. The exact host-side trigger is unverified.

Nsight Compute returned **ERR_NVGPUCTRPERM both before and after the restart**. See `nsight-after-restart/status.json` and its raw CSV. No hardware counter values are claimed. The serving processes have exited; both GPUs were idle with 1 MiB used at the final check.

For this r570 driver, the provider must enable profiling on the host or grant a supported profiling capability. Ask support: “Nsight Compute returns ERR_NVGPUCTRPERM inside my GPU pod. Can this host/container be configured to allow NVIDIA GPU performance counters?” Probe before renting another host. [NVIDIA permissions guidance](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Reproduce and preserve

The 40 GB container disk has no mounted persistent workspace volume. Results have been copied locally. Run one workload at a time after preparing the two isolated environments:

```bash
python -m scripts.run_remaining --requests 32 --out-dir results/h100-dual
```

The earlier quantization, FP8, compiler, edge and Whisper measurements remain in [the RTX 4090 report](../runpod/report.md). Earlier SGLang measurements in `results/serving/` used a different host and are not included in this matched H100 comparison.
