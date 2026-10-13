"""TensorRT 10 execution on PyTorch's CUDA context; fixed shapes, named I/O."""
from pathlib import Path


def require_cuda():
    import torch
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("An NVIDIA CUDA GPU is required; CPU results are not substituted")
    return torch


def build_engine(onnx_path, engine_path, precision="fp32", workspace_gib=2):
    import tensorrt as trt
    if int(trt.__version__.split('.')[0]) != 10:
        raise RuntimeError("This builder targets TensorRT 10.x; use the documented environment")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("ONNX parse failed:\n" + '\n'.join(str(parser.get_error(i))
                                                           for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.FP16)  # floating fallback; Q/DQ still fixes INT8 arithmetic
        # Explicit Q/DQ nodes specify calibrated INT8; do not set implicit INT8 flags.
        import onnx
        graph = onnx.load(str(onnx_path)).graph
        if not any(n.op_type == "QuantizeLinear" for n in graph.node):
            raise ValueError("INT8 requires an explicitly calibrated Q/DQ ONNX model")
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    path = Path(engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(blob))
    return path


class TensorRTRunner:
    def __init__(self, path):
        import tensorrt as trt
        self.torch = require_cuda()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {path}; rebuild on this GPU/software stack")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        dtype_map = {trt.float32: self.torch.float32, trt.float16: self.torch.float16,
                     trt.int32: self.torch.int32, trt.int64: self.torch.int64,
                     trt.bool: self.torch.bool, trt.int8: self.torch.int8}
        self.inputs, self.outputs = {}, {}
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError("Only fixed-shape engines are supported")
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError("Host/shape I/O tensors are not supported")
            spec = (shape, dtype_map[self.engine.get_tensor_dtype(name)])
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = spec
            else:
                self.outputs[name] = spec
                self.buffers[name] = self.torch.empty(shape, dtype=spec[1], device="cuda")
                self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(self, **inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Expected inputs {list(self.inputs)}, got {list(inputs)}")
        for name, value in inputs.items():
            shape, dtype = self.inputs[name]
            if tuple(value.shape) != shape or value.dtype != dtype or not value.is_cuda or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous CUDA {dtype} of shape {shape}")
            self.context.set_tensor_address(name, value.data_ptr())
        # The caller keeps input tensors alive until synchronization.
        stream = self.torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.buffers

    def inspect(self):
        import json
        import tensorrt as trt
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.context
        return json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
