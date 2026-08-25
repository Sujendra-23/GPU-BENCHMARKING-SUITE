# Base image ships CUDA, cuDNN, and the driver userspace libraries needed
# for torch/tensorrt to see the GPU inside the container.
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace
COPY harness/ ./harness/
COPY training/ ./training/
COPY kernels/ ./kernels/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Build the CUDA kernel binary at image-build time so the container is
# ready to benchmark immediately on `docker run`.
RUN nvcc -O3 -o /usr/local/bin/matmul_cuda kernels/cuda/matmul.cu

ENTRYPOINT ["python3", "harness/benchmark.py"]
