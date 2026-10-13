# RunPod: quantization, compilers, serving, edge and ASR

Run commands from the repository root. Run one GPU workload at a time. The
single-GPU commands below are implemented; the result files distinguish failed
runs from measurements. New measurements must be collected on the GPU, not
inferred from earlier results. See `results/runpod/` for this session's results.

## Existing pod and storage

This session connected to an RTX 4090, driver 580.159.04, Python 3.11.10,
CUDA toolkit 12.4, and a **20 GB container disk with no mounted workspace volume**.
Use the SSH command RunPod supplies in **Connect → SSH over exposed TCP**.
Do not commit the Jupyter token or SSH details. The runner binds HTTP services
to localhost; you do not need to expose a serving port for local benchmarks.

For a fresh pod, choose a full Ada/Hopper GPU (4090/L40S/H100), Python 3.11,
and a CUDA development template. Provision **at least 60 GB disk** if retaining
several serving environments; 80–100 GB is more comfortable. These are storage
allowances, not claims about GPU memory requirements. Mount a persistent volume
at `/workspace` if you need files to survive replacing the pod. Copy results
back before terminating a pod. [RunPod storage documentation](https://docs.runpod.io/pods/storage/types).

```bash
cd /workspace
# Or upload the working tree with SCP; the new files must be included.
# Clone the benchmark source and measured reports:
git clone https://github.com/Sujendra-23/GPU-BENCHMARKING-SUITE.git gpu-benchmarking-suite
cd gpu-benchmarking-suite
python3.11 -m venv .venv-gpu
.venv-gpu/bin/pip install --no-cache-dir -r requirements-gpu-extensions.txt
.venv-gpu/bin/python -m pip check
```

The pins pair PyTorch 2.8, Torch-TensorRT 2.8 and TensorRT 10.12. The PyTorch
wheel includes CUDA 12.8 libraries; use a compatible driver. The suite targets
TensorRT 10's weakly typed builder and explicitly rejects TensorRT 11.
[Torch-TensorRT installation](https://docs.pytorch.org/TensorRT/v2.8.0/getting_started/installation.html).

## 1. Quantization first

For a quick, public-data run, download
[Imagenette-160](https://github.com/fastai/imagenette), which contains ten ImageNet
classes. **Report its accuracy as Imagenette subset accuracy, not full ImageNet
accuracy.** Evaluation retains the original 1,000 output logits and maps class
folders to their original indices. Calibration uses train; accuracy uses val.

```bash
mkdir -p data
curl -fL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -o data/imagenette.tgz
tar -xzf data/imagenette.tgz -C data
.venv-gpu/bin/python -m training.export_tensorrt --suite --batch-size 1 \
  --calibration-dir data/imagenette2-160/train --eval-dir data/imagenette2-160/val \
  --class-index training/imagenette-class-index.json \
  --dataset-label 'Imagenette-160 validation; 10-class subset; 1000 logits' \
  --calibration-samples 512 --eval-samples 1000 \
  --engine-dir results/runpod/engines --out-dir results/runpod/quantization
```

Both ResNet-18 and ResNet-50 get FP32, FP16 and INT8 engines, identical images,
top-1/top-5 accuracy, delta in percentage points, warmed latency samples, speedup,
serialized bytes, size ratios, and layer inspection. FP32 disables TF32. INT8
uses GPU entropy calibration and symmetric per-channel ONNX Q/DQ. The stem
convolution and biases stay unquantized, avoiding an unsupported INT8 stem/MaxPool
fusion; floating fallback may use FP16/FP32. The layer reports let you inspect the actual
engine. Calibration image hashes and evaluation hashes are retained and checked
for overlap. The INT8 batch builder pads only the final batch. Accuracy excludes
padded examples.

For full ImageNet accuracy, use your organized ImageNet train/val roots with all
1,000 wnid folders and omit `--class-index`. A labeled subset requires your JSON
folder → original-class-index mapping. Increase samples for a more representative
accuracy estimate. Re-run with `--batch-size 8` into a separate output directory
for throughput; engines are shape-, GPU- and software-specific.

```bash
# Install into this environment only after quantization dependencies settle.
.venv-gpu/bin/pip install --no-cache-dir -r serving/requirements-vllm.txt
.venv-gpu/bin/python -m serving.run --backend vllm --quantization none --requests 64 \
  --out-dir results/runpod/serving
.venv-gpu/bin/python -m serving.run --backend vllm --quantization fp8 --requests 64 \
  --out-dir results/runpod/serving
```

Online FP8 uses the pinned Qwen2.5-0.5B-Instruct checkpoint. KV remains FP16;
this is weight/activation quantization, not FP8 KV. Keep launch logs as evidence
of the selected quantization. [vLLM FP8 documentation](https://docs.vllm.ai/en/v0.10.2/features/quantization/fp8.html).
The serving workload measures latency/throughput and sampled KV occupancy, not
language quality. Results have distinct names such as `vllm-fp8-tp1.json`.

## 2. Torch-TensorRT

```bash
.venv-gpu/bin/python -m compilers.torch_tensorrt_block \
  --models llama-block,resnet18,resnet50 --precision fp16 --batch-size 1 \
  --out-dir results/runpod/compiler 2>&1 | tee results/runpod/compiler-console.log
```

All four paths share weights and inputs: eager, Inductor, Torch-TensorRT Dynamo,
and a serialized native ONNX→TensorRT engine. Each output is checked against
eager before timing. Build/first-call time is separate. Warmed timing includes
host dispatch and CUDA synchronization without CUDA graphs. JSON records graph
break counts/reasons and compilation counters; the partitioned graph and module
list identify TensorRT segments and PyTorch fallbacks. A run with no TRT runtime
modules fails. [Torch-TensorRT partitioning](https://docs.pytorch.org/TensorRT/user_guide/compilation/dynamo_export.html).

The LLaMA-style block has RMSNorm, RoPE, causal attention and SwiGLU. Its seeded
random weights and default hidden size 512 exercise compilation; they are not a
pretrained LLaMA accuracy result. Use `--hidden 2048 --heads 16 --intermediate 5504`
for a larger shape. There is no KV cache or backward pass.

## 3. TensorRT-LLM next to vLLM/SGLang

**Do not install TensorRT-LLM into the compiler/vLLM environment.** Pick a
release compatible with the host driver. CUDA 12.x / r570 hosts use the new
`serving/requirements-tensorrt-llm-cu12.txt` environment: TensorRT-LLM 1.0.0,
PyTorch 2.7.1+cu128, TensorRT 10.11 and Transformers 4.53.1. The default
PyPI PyTorch wheel uses CUDA 12.6, which failed at DeepGEMM import with
`undefined symbol: cudaLibraryGetKernel` on the H100 pod; the requirements
explicitly select the CUDA 12.8 builds from the PyTorch wheel index.
NVIDIA's 1.0.0 wheel
index has Python 3.10/3.12 builds, but no Python 3.11 build; use Python 3.10
on Ubuntu 22.04. CUDA 13 / r580 hosts may use the separate 1.2.0 requirements,
which require PyTorch >=2.9.1 and TensorRT 10.14. These conflict with the
compiler environment's pins.
[NVIDIA CUDA 12 release installation](https://nvidia.github.io/TensorRT-LLM/1.0.0/installation/linux.html).

```bash
apt-get update && apt-get install -y libnuma1 libopenmpi-dev python3.10-venv python3.10-dev
python3.10 -m venv .venv-trtllm-cu12
.venv-trtllm-cu12/bin/pip install --no-cache-dir -r serving/requirements-tensorrt-llm-cu12.txt
.venv-trtllm-cu12/bin/python -m pip check
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  .venv-trtllm-cu12/bin/python -m serving.run --backend tensorrt-llm --requests 32 \
  --startup-timeout 1200 --out-dir results/h100-dual/serving
```

The launcher uses the configuration-file alias accepted by both releases and
pins both model and tokenizer revisions inside that file. It records the
installed TensorRT-LLM version and CUDA_VISIBLE_DEVICES. The protobuf environment
setting supports the legacy etcd3 imports in the CUDA 12 release.

This uses **TensorRT-LLM's PyTorch executor**, explicitly recorded in the report,
not a serialized TensorRT-LLM engine. It still benchmarks the TensorRT-LLM serving
runtime. The runner supplies FP16 dtype, disables block reuse/CUDA graphs, sets
the same token capacity, and enables iteration stats. TensorRT-LLM's `/metrics`
is JSON; cache fractions are derived from real used/max block counts, retaining
raw counts. Missing counters fail rather than producing invented memory results.
[TensorRT-LLM server and metrics](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).

For release 1.0.0, `serving.trtllm_server` adapts only the HTTP metrics reader:
it returns real iteration stats within a 20 ms drain window and retains already
collected stats if the upstream empty-deque bug raises `IndexError`. The original
endpoint could wait for a two-second gap between stats and time out during load.
Model execution and the stats producer are unchanged. The launch report records
this adapter; newer releases use their original metrics endpoint.

SGLang remains available in its own environment using
`serving/requirements-sglang.txt`; run `python -m serving.run --backend sglang`.
The same completion client uses exact token-ID prompts and requires actual
server prompt/output counts to match the requested workload.

## 4. Edge profile

The batch-1 engines above are prerequisites. The cap is a requested whole-board
limit; it may be refused by the provider. No Jetson behavior is inferred.

```bash
.venv-gpu/bin/python -m harness.edge_profile --model resnet18 --power-limit 200 \
  --engine-dir results/runpod/engines --out-dir results/runpod/edge
```

The label is **“edge-style, not Jetson”**. Baseline and second run have separate
CSV/logs and whole-board power/utilization/clock/temperature samples. A refused
cap produces `cap-refused-uncapped`, not a purported power-capped measurement.
`profile.json` records the request, observed limit and restoration verification.
Use the full GPU UUID; the subprocess receives that UUID in CUDA_VISIBLE_DEVICES.
The runner attempts to restore the original limit in `finally`. If forcibly
killed or disconnected at an unfortunate moment, check `nvidia-smi -q -d POWER`
and restore the recorded original limit manually.

## 5. Whisper-tiny ASR

```bash
.venv-gpu/bin/python -m harness.prepare_whisper_data --samples 8
.venv-gpu/bin/python -m harness.benchmark --model whisper-tiny --batch-sizes 1 \
  --precisions fp32 --frameworks torch,onnx,tensorrt,torch.compile \
  --audio-manifest data/whisper/manifest.jsonl --audio-samples 8 \
  --warmup 3 --iterations 20 --out results/runpod/whisper.csv \
  --asr-out-dir results/runpod/whisper
```

The public LibriSpeech dummy fixture is a smoke/functional check, not a
representative ASR evaluation set. Supply your JSONL (`{"audio":"clip.wav",
"text":"reference transcript"}`; paths relative to manifest) for meaningful WER.
Clips must be <=30 seconds; resampling to 16 kHz and CPU mel preprocessing are
outside timed regions. The report retains transcripts, normalized word error
counts, WER, per-batch latency and RTF. English greedy decoding recomputes a
fixed padded decoder sequence **without a KV cache**. All runtimes apply the same
suppression rules. Encoder and first-step logits are checked against eager (FP32 atol/rtol 1e-3,
FP16 3e-2, with maximum errors retained). TF32 is disabled in all timed FP32
paths, including ONNX Runtime; floating-point accumulation order can still differ.
Token-limit truncation is flagged. Use `--precisions fp32,fp16` to sweep both.
Export/build/compile time is excluded from warmed transcription latency.

A prepared quick run of the single-GPU stages is:

```bash
.venv-gpu/bin/python -m scripts.runpod_single_gpu --quick
# Failure logs are retained and later stages still run; see results/runpod/status.json.
```

## 6. Separate two-GPU rental — lowest priority

Rent **two matching visible GPUs in one pod**, with sufficient disk and a driver
compatible with the pinned vLLM wheel. Prefer a provider exposing GPU topology.
Check `nvidia-smi -L` and `nvidia-smi topo -m`. Do not assume NVLink (4090 has none).
Use the same environment and workload for TP1 and TP2 on this new host:

```bash
python3.11 -m venv .venv-vllm
.venv-vllm/bin/pip install --no-cache-dir -r serving/requirements-vllm.txt
.venv-vllm/bin/python -m pip check
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python -m serving.run --backend vllm \
  --tensor-parallel-size 1 --out-dir results/tp/serving
CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/python -m serving.run --backend vllm \
  --tensor-parallel-size 2 --out-dir results/tp/serving
```

With both isolated serving environments prepared, run the remaining phases
sequentially with matching 32-request workloads:

```bash
nvidia-smi -L  # Must show two working GPUs, not an NVML error.
python -m scripts.run_remaining --requests 32 --out-dir results/h100-dual
# To retry a single phase: --stages trtllm (or tp1 or tp2).
```

Qwen has two KV heads. TP2 shards them without replication; the runner halves
the per-rank byte budget, keeping the aggregate KV payload budget constant.
Gauge labels are retained, never summed. The peak byte estimate is per rank,
not aggregate GPU memory. For this tiny model, communication may outweigh any
speedup; measure rather than assume. The runner refuses fewer than two GPUs.

## 7. Counter-enabled host or bare metal

Try the existing host first; no additional rental is needed if the probe passes.
For this pod, Nsight is at `/opt/nvidia/nsight-compute/2024.1.1/ncu`.

```bash
export PATH=/opt/nvidia/nsight-compute/2024.1.1:$PATH
python -m harness.nsight_counters --probe-only --out-dir results/nsight/probe
python -m harness.nsight_counters --out-dir results/nsight/counters
```

Run profiling separately from latency runs. Preserve CSVs and `.ncu-rep` files.
If `ERR_NVGPUCTRPERM` appears, ask the provider whether its **host driver permits
performance counters for your container** before renting. Root inside the pod
is not enough by itself. A dedicated/bare-metal host under your control can
configure this; do not attempt to change shared-provider driver policy.
[NVIDIA counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
For an r570 host, ask the provider to enable profiling in its host driver or
grant the container a supported profiling capability (`CAP_PERFMON` or
`CAP_SYS_ADMIN`). An exact support request is: “Nsight Compute returns
ERR_NVGPUCTRPERM inside my GPU pod. Can this host/container be configured to
allow NVIDIA GPU performance counters?” Verify with the probe before starting
another rental for profiling.
Collect SpeedOfLight, Occupancy, MemoryWorkloadAnalysis and InstructionStats for
CUDA/Triton softmax/attention; interpret actual counters before making SM,
DRAM, occupancy or tensor-instruction claims.

## Preserve the current artifacts

The run report is `results/runpod/report.md`. Source, logs and JSON reports are
committed with the benchmark source and were also uploaded to the pod. The engine files are specific to the recorded GPU/software stack. The
session's local backup lives under `engines/runpod/`; use the matching files
under the pod's `results/runpod/engines/` for the inference harness. Native
compiler and Whisper engines have their own model-specific directories.

To retrieve files from any new pod using its supplied direct SSH details:

```bash
# Run on your local machine; substitute the host/port from RunPod Connect.
scp -P PORT -i ~/.ssh/id_ed25519 root@HOST:/workspace/gpu-benchmarking-suite/runpod-results.tgz .
```

The later H100 pod has two GPUs and NVLink; its probe and setup evidence is in
`results/h100-dual/`. TensorRT-LLM 1.0.0 and matched vLLM TP1/TP2 completed with
zero failed requests, exact token counts and valid KV metrics. See
[the measured H100 report](../results/h100-dual/report.md). Before serving could
start, the container lost GPU access:
`nvidia-smi` reported `Failed to initialize NVML: Unknown Error`, and opening
`/dev/nvidiactl` returned `EPERM`, including with the original environment.
Restarting through RunPod restored access and preserved the environments.
Nsight probes before and after restart independently returned
`ERR_NVGPUCTRPERM`; hardware counters still require provider-enabled access.

If this GPU-access error occurs, first back up results and source. Restart the
pod through RunPod, then recheck `nvidia-smi -L` and PyTorch CUDA visibility.
The observed error is consistent with NVIDIA's documented container device
access failure, but the exact host-side trigger has not been verified. If it
persists, ask RunPod support to restore the container's GPU device access.
[NVIDIA Container Toolkit troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error).

No additional rentals were created by the benchmark runner. The final reports,
logs and environment manifests were copied locally; the serving processes have
exited. The pod remains rented until you stop or terminate it in RunPod.
