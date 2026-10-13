"""Fixed-shape ONNX Runtime CUDA execution with device-resident named I/O."""


class ONNXRunner:
    def __init__(self, path, output_shapes):
        import numpy as np
        import onnxruntime as ort
        from harness.tensorrt_utils import require_cuda
        self.torch = require_cuda()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('onnxruntime-gpu with CUDAExecutionProvider is required')
        self.session = ort.InferenceSession(str(path), providers=[('CUDAExecutionProvider', {
            'device_id': self.torch.cuda.current_device(), 'use_tf32': '0',
            'user_compute_stream': str(self.torch.cuda.current_stream().cuda_stream)})])
        if self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('ONNX Runtime failed to initialize its CUDA provider')
        types = {'tensor(float)': (np.float32, self.torch.float32),
                 'tensor(float16)': (np.float16, self.torch.float16),
                 'tensor(int64)': (np.int64, self.torch.int64),
                 'tensor(int32)': (np.int32, self.torch.int32)}
        self.specs = {n.name: types[n.type] for n in self.session.get_inputs()}
        self.buffers, self.output_types = {}, {}
        for node in self.session.get_outputs():
            np_type, dtype = types[node.type]
            self.output_types[node.name] = np_type
            self.buffers[node.name] = self.torch.empty(output_shapes[node.name], dtype=dtype, device='cuda')
        self.binding = self.session.io_binding()
        for name, output in self.buffers.items():
            self.binding.bind_output(name, 'cuda', output.device.index, self.output_types[name],
                                     tuple(output.shape), output.data_ptr())

    def __call__(self, **inputs):
        if set(inputs) != set(self.specs):
            raise ValueError(f'ONNX inputs: expected {list(self.specs)}, got {list(inputs)}')
        for name, value in inputs.items():
            np_type, dtype = self.specs[name]
            if not value.is_cuda or not value.is_contiguous() or value.dtype != dtype:
                raise ValueError(f'{name} must be contiguous CUDA {dtype}')
            self.binding.bind_input(name, 'cuda', value.device.index, np_type, tuple(value.shape), value.data_ptr())
        self.session.run_with_iobinding(self.binding)
        return self.buffers
